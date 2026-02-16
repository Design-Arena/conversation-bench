"""Text-based pipeline for synchronous LLM services.

This pipeline works with text-in/text-out LLM services:
- OpenAI (GPT-4o, GPT-4.1, etc.)
- Anthropic (Claude Sonnet, Claude Haiku, etc.)
- Google (Gemini Flash, etc.)
- AWS Bedrock (Claude, Llama, etc.)
- OpenRouter (various models)

Pipeline: UserAggregator → LLM → ToolCallRecorder → AssistantAggregator → NextTurn
"""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTimestampFrame,
    LLMRunFrame,
    MetricsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from audio_arena.pipelines.base import BasePipeline
from audio_arena.processors.tool_call_recorder import ToolCallRecorder


class NextTurn(FrameProcessor):
    """Frame processor that detects end-of-turn and handles metrics.

    Watches for LLMContextAssistantTimestampFrame which signals that the
    assistant's response is complete and has been added to context.
    """

    def __init__(self, end_of_turn_callback, metrics_callback):
        super().__init__()
        self.end_of_turn_callback = end_of_turn_callback
        self.metrics_callback = metrics_callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            self.metrics_callback(frame)

        # Treat assistant timestamp frame as end-of-turn marker
        if isinstance(frame, LLMContextAssistantTimestampFrame):
            logger.info("EOT (timestamp)")
            await self.end_of_turn_callback()


class TextPipeline(BasePipeline):
    """Pipeline for text-based (synchronous) LLM services.

    This is the simplest pipeline type:
    1. User message is added to context
    2. LLMRunFrame triggers the LLM
    3. LLM responds with text
    4. Context aggregator captures response
    5. NextTurn detects end-of-turn and advances
    """

    requires_service = True

    def __init__(self, benchmark):
        super().__init__(benchmark)
        self.context_aggregator = None
        self.last_msg_idx = 0

    def _setup_context(self) -> None:
        """Create LLMContext with system prompt, tools, and first user message."""
        # Get system instruction from benchmark
        system_instruction = getattr(self.benchmark, "system_instruction", "")

        # Initial messages: system + first user turn
        first_turn = self._get_current_turn()
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": first_turn["input"]},
        ]

        # Get tools schema from benchmark
        tools = getattr(self.benchmark, "tools_schema", None)

        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)
        self.last_msg_idx = len(messages)

    def _setup_llm(self) -> None:
        """Register the function handler for all tools."""
        self.llm.register_function(None, self._function_catchall)

    def _build_task(self) -> None:
        """Build the pipeline with context aggregators and turn detector."""

        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        # Create the end-of-turn handler
        async def end_of_turn():
            if self.done:
                return

            # Extract assistant text from context
            # context_aggregator.assistant() has already added the message
            msgs = self.context.get_messages()
            assistant_text = ""
            if msgs and msgs[-1].get("role") == "assistant":
                content = msgs[-1].get("content", "")
                assistant_text = content if isinstance(content, str) else ""

            await self._on_turn_end(assistant_text)

        next_turn = NextTurn(end_of_turn, self._handle_metrics)

        pipeline = Pipeline(
            [
                self.context_aggregator.user(),
                self.llm,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.context_aggregator.assistant(),
                next_turn,
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=45,
            idle_timeout_frames=(MetricsFrame,),
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

    async def _queue_first_turn(self) -> None:
        """Queue LLMRunFrame to start the first turn."""
        # The first user message is already in context from _setup_context
        await self.task.queue_frames([LLMRunFrame()])

    async def _queue_next_turn(self) -> None:
        """Add next user message to context and trigger LLM."""
        turn = self._get_current_turn()
        self.context.add_messages([{"role": "user", "content": turn["input"]}])
        self.last_msg_idx = len(self.context.get_messages())
        await self.task.queue_frames([LLMRunFrame()])
