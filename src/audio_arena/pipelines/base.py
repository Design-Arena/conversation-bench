"""Base pipeline class for multi-turn evaluation.

The pipeline owns EVERYTHING - the CLI/runner just calls pipeline.run().
Each pipeline type (text, realtime, nova-sonic) handles its own specifics.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams

from audio_arena.recording.transcript_recorder import TranscriptRecorder


class BasePipeline(ABC):
    """Base class for all pipelines. Owns all execution state and logic.

    The pipeline is responsible for:
    1. Creating and configuring the LLM service
    2. Setting up context with system prompt and tools
    3. Building the pipeline with all processors
    4. Managing turn flow (queuing turns, detecting end-of-turn)
    5. Recording transcripts and metrics

    Subclasses implement the abstract methods to customize behavior.
    """

    # Set to False for pipelines that create their own LLM (e.g., Nova Sonic)
    requires_service = True

    def __init__(self, benchmark):
        """Initialize the pipeline.

        Args:
            benchmark: A BenchmarkConfig instance with turns, tools, and system instruction.
        """
        self.benchmark = benchmark
        self.turns = benchmark.turns
        self.turn_idx = 0
        self.done = False
        self.recorder: Optional[TranscriptRecorder] = None
        self.task: Optional[PipelineTask] = None
        self.context: Optional[LLMContext] = None
        self.llm: Optional[FrameProcessor] = None
        self.model_name: Optional[str] = None
        self.service_name: Optional[str] = None
        self._turn_indices: Optional[List[int]] = None
        # Golden turns to inject as context before the target turn (single-step rehydration)
        self._rehydration_turns: Optional[List[Dict[str, Any]]] = None
        # Track tool calls to detect duplicates within a turn
        self._seen_tool_calls: set = set()
        # Track tool_call_ids that are duplicates (for filtering in ToolCallRecorder)
        self._duplicate_tool_call_ids: set = set()
        # Track which response index we're on for multi-step tool chains
        self._tool_response_idx: int = 0
        # Last tool result (for explicit delivery to GPT/Grok Realtime APIs; see docstring below)
        self._last_tool_result: Optional[Dict[str, Any]] = None

    @staticmethod
    def build_rehydration_history(
        golden_turns: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Build rehydration context from golden turns for single-step evaluation.

        Converts golden turn definitions into two formats:
        - A messages list (user/assistant pairs) for text pipeline LLMContext injection.
        - A formatted instruction string for realtime pipeline system-instruction enrichment.

        Args:
            golden_turns: The benchmark turns list sliced to ``turns[0:target_turn_idx]``.

        Returns:
            ``(messages, instruction_string)`` tuple.
        """
        messages: List[Dict[str, Any]] = []
        lines = [
            "\n\n--- CONVERSATION HISTORY (GOLDEN) ---",
            "The following is the conversation so far. The user's name, "
            "preferences, and any actions you have taken (tool calls, "
            "registrations, schedule changes, etc.) are still in effect. "
            "Continue naturally.",
            "",
        ]

        for i, turn in enumerate(golden_turns):
            user_input = turn.get("input", "")
            golden_text = turn.get("golden_text", "")
            fc = turn.get("required_function_call")
            fc_response = turn.get("function_call_response")

            messages.append({"role": "user", "content": user_input})
            lines.append(f"User: {user_input}")

            if fc is not None:
                calls = fc if isinstance(fc, list) else [fc]
                responses = (
                    fc_response
                    if isinstance(fc_response, list)
                    else [fc_response] if fc_response is not None else []
                )
                for j, call in enumerate(calls):
                    lines.append(
                        f"  [Tool call: {call['name']}({json.dumps(call.get('args', {}))})]"
                    )
                    if j < len(responses):
                        lines.append(f"  [Tool result: {json.dumps(responses[j])}]")

            messages.append({"role": "assistant", "content": golden_text})
            lines.append(f"Assistant: {golden_text}")
            lines.append("")

        instruction_string = "\n".join(lines)
        return messages, instruction_string

    @property
    def effective_turns(self) -> List[dict]:
        """Get the turns to run (filtered by turn_indices if set)."""
        if self._turn_indices is not None:
            return [self.turns[i] for i in self._turn_indices if i < len(self.turns)]
        return self.turns

    async def run(
        self,
        recorder: TranscriptRecorder,
        model: str,
        service_class: Optional[type] = None,
        service_name: Optional[str] = None,
        turn_indices: Optional[List[int]] = None,
        rehydration_turns: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Run the complete benchmark. Pipeline handles everything internally.

        Args:
            recorder: TranscriptRecorder for saving results.
            model: Model name/identifier.
            service_class: LLM service class (required unless pipeline sets requires_service=False).
            service_name: Service name/alias (e.g., "openai", "openrouter").
            turn_indices: Optional list of turn indices to run (for debugging).
            rehydration_turns: Optional golden turns to inject as prior context.
                When set, the pipeline runs in single-step rehydration mode: the golden
                history is injected into the model context, and only the target turn(s)
                specified by ``turn_indices`` are executed live.
        """
        self.recorder = recorder
        self.model_name = model
        self.service_name = service_name  # Store for use in _create_llm overrides
        self._turn_indices = turn_indices
        self._rehydration_turns = rehydration_turns

        # Create LLM service
        self.llm = self._create_llm(service_class, model)

        # Setup (pipeline-specific)
        self._setup_context()
        self._setup_llm()
        self._build_task()

        # Initialize first turn BEFORE queueing
        self.recorder.start_turn(self._get_actual_turn_index(0))

        # Queue first turn and run
        await self._queue_first_turn()
        runner = PipelineRunner(handle_sigint=True)
        await runner.run(self.task)

    def _get_actual_turn_index(self, effective_index: int) -> int:
        """Convert effective turn index to actual turn index."""
        if self._turn_indices is not None:
            return self._turn_indices[effective_index]
        return effective_index

    def _get_current_turn(self) -> dict:
        """Get the current turn data."""
        return self.effective_turns[self.turn_idx]

    def _create_llm(
        self, service_class: Optional[type], model: str
    ) -> FrameProcessor:
        """Create LLM service. Override for pipelines that create their own.

        Args:
            service_class: LLM service class to instantiate.
            model: Model name/identifier.

        Returns:
            Configured LLM service instance.

        Note:
            Subclasses can access self.service_name if needed for service-specific config.
        """
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        # Build kwargs with API keys based on service class name
        kwargs: Dict[str, Any] = {"model": model}
        class_name = service_class.__name__
        model_lower = model.lower()
        service_name_lower = (self.service_name or "").lower()

        # Handle OpenRouter (uses OpenAI-compatible API with different base URL and API key)
        if service_name_lower == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
            logger.info(f"Using OpenRouter with base_url={kwargs['base_url']}")
            return service_class(**kwargs)

        if "Anthropic" in class_name:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "Groq" in class_name:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "Cerebras" in class_name:
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise EnvironmentError("CEREBRAS_API_KEY environment variable is required")
            kwargs["api_key"] = api_key
        elif "OpenAI" in class_name:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Configure gpt-5 series models: set reasoning effort and priority tier
            if model_lower.startswith("gpt-5"):
                from pipecat.services.openai.llm import OpenAILLMService
                # gpt-5.1 and gpt-5.2 use "none", other gpt-5 models use "minimal"
                if model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt-5.2"):
                    reasoning_effort = "none"
                else:
                    reasoning_effort = "minimal"
                kwargs["params"] = OpenAILLMService.InputParams(
                    service_tier="priority",
                    extra={"reasoning_effort": reasoning_effort},
                )
                logger.info(f"Configured {model} with reasoning_effort={reasoning_effort}, service_tier=priority")

        elif "Google" in class_name or "Gemini" in class_name:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY environment variable is required")
            kwargs["api_key"] = api_key

            # Configure gemini-3 series models: use minimal thinking
            if "gemini-3" in model_lower:
                from google.genai import types
                from pipecat.services.google.llm import GoogleLLMService
                kwargs["params"] = GoogleLLMService.InputParams(
                    extra={
                        "thinking_config": types.ThinkingConfig(
                            thinking_level="MINIMAL",
                            include_thoughts=True,
                        )
                    }
                )
                logger.info(f"Configured {model} with thinking_level=MINIMAL")

        elif "Bedrock" in class_name:
            # AWS Bedrock uses AWS credentials from environment
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            if not (access_key and secret_key):
                raise EnvironmentError(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required"
                )
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
            session_token = os.getenv("AWS_SESSION_TOKEN")
            if session_token:
                kwargs["aws_session_token"] = session_token
            region = os.getenv("AWS_REGION", "us-east-1")
            kwargs["region"] = region

        return service_class(**kwargs)

    async def _on_turn_end(self, assistant_text: str) -> None:
        """Called when assistant finishes. Base handles common logic.

        Args:
            assistant_text: The assistant's response text.
        """
        if self.done:
            return

        # Get the actual turn index for recording
        actual_turn_idx = self._get_actual_turn_index(self.turn_idx)

        # Record turn (common)
        self.recorder.write_turn(
            user_text=self.effective_turns[self.turn_idx].get("input", ""),
            assistant_text=assistant_text,
        )

        # Advance (common)
        self.turn_idx += 1

        # Reset tool call tracking for the new turn
        self._seen_tool_calls.clear()
        self._duplicate_tool_call_ids.clear()
        self._tool_response_idx = 0

        if self.turn_idx < len(self.effective_turns):
            # Start next turn
            actual_next_idx = self._get_actual_turn_index(self.turn_idx)
            self.recorder.start_turn(actual_next_idx)
            await self._queue_next_turn()
        else:
            # All turns complete
            logger.info("Conversation complete")
            self.done = True
            self.recorder.write_summary()
            await self.task.cancel()

    def _handle_metrics(self, frame: MetricsFrame) -> None:
        """Common metrics handling."""
        for md in frame.data:
            if isinstance(md, LLMUsageMetricsData):
                self.recorder.record_usage_metrics(md.value, getattr(md, "model", None))
            elif isinstance(md, TTFBMetricsData):
                self.recorder.record_ttfb(md.value)

    async def _function_catchall(self, params: FunctionCallParams) -> None:
        """Common function handler: returns result (from turn data or default), handles end_session.

        Tool call recording is handled by ToolCallRecorder in the pipeline. This handler
        returns the result and handles the special end_session tool.

        Duplicate tool calls (same function + args) are detected and skipped to prevent
        context pollution.

        Explicit results for APIs (only GPT / Grok Realtime):
        Only the OpenAI Realtime protocol (used by OpenAI and xAI for realtime/voice)
        requires the client to push the tool result over the WebSocket (conversation.item.create
        with function_call_output, then response.create). Other providers (text/chat,
        Gemini Live, Ultravox, Nova Sonic) use request/response or other flows where
        Pipecat delivers the handler result automatically; no extra send is needed.
        For GPT and Grok we set self._last_tool_result and pass a getter into their
        services so they send the actual payload (e.g. from the turn's function_call_response)
        instead of a hardcoded {"status": "success"}.

        Ordering guarantee (tool result before model speaks):
        Pipecat awaits this handler before injecting the result and letting the model
        continue. The model only gets to generate speech/text AFTER result_callback() is
        called. If you add real async work (e.g. API calls) to compute the tool response,
        complete that work first, then call result_callback(result). Do not call
        result_callback from a background task or before the result is ready.
        """
        # Create a key for duplicate detection (function_name + args)
        call_key = (params.function_name, str(params.arguments or {}))

        # Check for duplicate tool call
        if call_key in self._seen_tool_calls:
            tool_call_id = getattr(params, 'tool_call_id', None)
            logger.warning(
                f"Skipping duplicate tool call: {params.function_name} "
                f"(tool_call_id={tool_call_id})"
            )
            # Track this tool_call_id as a duplicate so ToolCallRecorder can filter it
            if tool_call_id:
                self._duplicate_tool_call_ids.add(tool_call_id)
            # Return a result to satisfy the API, but mark it as skipped
            skip_result = {"status": "duplicate_skipped"}
            self._last_tool_result = skip_result
            await params.result_callback(skip_result)
            return

        # Track this call
        self._seen_tool_calls.add(call_key)

        # Check if the current turn has a custom function_call_response
        # Supports both single responses and lists of responses for multi-step tool chains
        result = {"status": "success"}
        if self.turn_idx < len(self.effective_turns):
            current_turn = self.effective_turns[self.turn_idx]
            custom_response = current_turn.get("function_call_response")
            if custom_response is not None:
                if isinstance(custom_response, list):
                    # Multi-step tool chain: use response at current index
                    if self._tool_response_idx < len(custom_response):
                        result = custom_response[self._tool_response_idx]
                        self._tool_response_idx += 1
                    else:
                        logger.warning(
                            f"Tool response index {self._tool_response_idx} exceeds "
                            f"available responses ({len(custom_response)}) - using default"
                        )
                else:
                    # Single response (backward compatible)
                    result = custom_response
        self._last_tool_result = result
        await params.result_callback(result)

        # end_session tool: gracefully terminate the run
        if params.function_name == "end_session":
            logger.info("end_session tool called - gracefully ending run")
            self.done = True
            # Small delay to let tool call frames propagate through ToolCallRecorder
            await asyncio.sleep(0.05)
            # Write the final turn (assistant response may be empty since it's just a tool call)
            if self.turn_idx < len(self.effective_turns):
                self.recorder.write_turn(
                    user_text=self.effective_turns[self.turn_idx].get("input", ""),
                    assistant_text="",
                )
            self.recorder.write_summary()
            # Cancel the pipeline task to exit cleanly
            await self.task.cancel()

    # ---- Abstract methods (pipeline-specific) ----

    @abstractmethod
    def _setup_context(self) -> None:
        """Create LLMContext with system prompt and tools."""
        pass

    @abstractmethod
    def _setup_llm(self) -> None:
        """Configure LLM (register functions, set callbacks)."""
        pass

    @abstractmethod
    def _build_task(self) -> None:
        """Build Pipeline and PipelineTask with all processors."""
        pass

    @abstractmethod
    async def _queue_first_turn(self) -> None:
        """Queue the first turn to start the conversation."""
        pass

    @abstractmethod
    async def _queue_next_turn(self) -> None:
        """Queue the next turn (called from _on_turn_end)."""
        pass
