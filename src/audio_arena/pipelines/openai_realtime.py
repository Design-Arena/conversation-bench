"""OpenAI Realtime with explicit tool result delivery.

Only the OpenAI Realtime protocol (and xAI's compatible one) requires the client to
push the tool result over the WebSocket—conversation.item.create (function_call_output)
then response.create. Other providers (text, Gemini Live, Ultravox, Nova Sonic) use
request/response or other flows where Pipecat delivers the handler result without
this extra step.

We subclass Pipecat's OpenAIRealtimeLLMService to send the function call output
ourselves so the audio model only continues (and speaks) after receiving the tool
result in context.

Explicit results for APIs (GPT / Grok only):
  We send the actual result from our pipeline handler, not a hardcoded
  {"status": "success"}. The pipeline sets _last_tool_result in _function_catchall
  and passes a getter (get_last_tool_result) into this service so we send that
  payload—e.g. per-turn function_call_response—to the API.

Why the constructor takes get_last_tool_result (constructor diff vs base):
  The pipeline and the LLM service are different objects. The handler that knows
  the real result (_function_catchall) runs on the pipeline and sets
  pipeline._last_tool_result. When we send the WebSocket event we're inside the
  *service* (e.g. in _handle_evt_function_call_arguments_done); the service has
  no reference to the pipeline, so it can't read pipeline._last_tool_result
  directly.   So when the pipeline *creates* the service it passes a getter,
  e.g. lambda: getattr(self, "_last_tool_result", {"status": "success"}),
  that closes over the pipeline. Later, when the service sends the event it
  calls get_last_tool_result(); that runs the lambda, which reads the pipeline's
  _last_tool_result and returns it.

  We could hardcode one payload in the service (e.g. {"status": "success"}), but
  we can't hardcode *per call* here—the service has no turn index or benchmark
  data; only the pipeline does. So we have to use the getter to pull the
  pipeline's result into the service when we send the event.

  How we know it's the right result for this call: we call the getter only after
  run_function_calls() returns. That await runs our handler (_function_catchall)
  for this one call; the handler sets _last_tool_result and then returns.
  Execution is sequential per event—no other tool call runs in between.

  Deferred response.create:
  The OpenAI Realtime API fires response.function_call_arguments.done *during*
  the active response (before response.done). Sending response.create at that
  point triggers "conversation_already_has_active_response" and kills the
  websocket.

  Additionally, Pipecat's base class has a context-update path:
  run_function_calls() → context update → _process_completed_function_calls()
  → _send_tool_result() + _create_response(). This also sends response.create
  during the active response.

  We solve both problems by:
  1. Completely overriding _handle_evt_function_call_arguments_done (no super())
     to run the tool ourselves and send our own function_call_output
  2. Pre-marking the tool_call_id as completed so the base class's
     _process_completed_function_calls() skips it
  3. Deferring response.create to _handle_evt_response_done, which fires
     after the response is complete and there is no active response
"""

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Optional

from loguru import logger

from pipecat.frames.frames import LLMFullResponseStartFrame
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class ReconnectOnDisconnectMixin:
    """Mixin for OpenAI-protocol LLM services that auto-reconnect on unexpected WS close.

    Provides:
    - ``_init_reconnection_callbacks(on_reconnecting, on_reconnected)`` — call from ``__init__``
    - ``_reconnect_on_disconnect()`` — reopen the WebSocket and fire callbacks
    - ``_handle_ws_close()`` — post-loop check; call at the end of ``_receive_task_handler``

    Used by both ``OpenAIRealtimeLLMServiceExplicitToolResult`` and ``XAIRealtimeLLMService``.
    """

    _on_reconnecting: Optional[Callable[[], None]]
    _on_reconnected: Optional[Callable[[], None]]

    def _init_reconnection_callbacks(
        self,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
    ):
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected

    async def _reconnect_on_disconnect(self):
        """Reconnect after an unexpected WebSocket disconnection.

        Opens a new WebSocket session and signals the pipeline to retry
        the current turn. Conversation history is enriched into the system
        instructions by the on_reconnecting callback before the new session
        starts.
        """
        if self._on_reconnecting:
            try:
                self._on_reconnecting()
            except Exception as e:
                logger.warning(f"Error in on_reconnecting callback: {e}")

        self._api_session_ready = False
        old_ws = self._websocket
        self._websocket = None

        try:
            if old_ws:
                await old_ws.close()
        except Exception:
            pass

        logger.info("Establishing new WebSocket connection...")
        await self._connect()

        for _ in range(100):
            if self._api_session_ready:
                break
            await asyncio.sleep(0.1)

        if self._api_session_ready:
            logger.info("Reconnection successful, session ready")
            if self._on_reconnected:
                try:
                    self._on_reconnected()
                except Exception as e:
                    logger.warning(f"Error in on_reconnected callback: {e}")
        else:
            logger.error("Reconnection failed — session not ready after 10s timeout")

    async def _handle_ws_close(self):
        """Check WebSocket close code after the receive loop exits.

        Call this at the end of ``_receive_task_handler()`` to detect
        unexpected disconnections and trigger automatic reconnection.
        """
        if getattr(self, '_disconnecting', False):
            return

        close_code = getattr(self._websocket, 'close_code', None) if self._websocket else None
        close_reason = getattr(self._websocket, 'close_reason', '') if self._websocket else ''

        if close_code is not None and close_code != 1000:
            logger.warning(
                f"WebSocket closed unexpectedly "
                f"(code={close_code}, reason={close_reason}), reconnecting..."
            )
            await self._reconnect_on_disconnect()
        elif close_code is not None:
            logger.info(f"WebSocket closed normally (code={close_code})")


class OpenAIRealtimeLLMServiceExplicitToolResult(ReconnectOnDisconnectMixin, OpenAIRealtimeLLMService):
    """OpenAI Realtime service that explicitly sends tool results to the API.

    Completely takes over function call handling to avoid the
    "conversation_already_has_active_response" error that occurs when
    response.create is sent during an active response.

    Also detects unexpected WebSocket disconnections and reconnects
    automatically via ``ReconnectOnDisconnectMixin``.

    Flow:
    1. response.function_call_arguments.done fires (during active response)
    2. We run the tool handler ourselves (run_function_calls)
    3. We pre-mark the call as completed to prevent base class auto-send
    4. We send conversation.item.create (function_call_output) with our result
    5. response.done fires (response is now complete)
    6. We send response.create to trigger the model to continue
    """

    def __init__(
        self,
        get_last_tool_result: Optional[Callable[[], dict]] = None,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        enable_manual_rehydrate_response_input: bool = False,
        rehydration_input_prefix: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._get_last_tool_result = get_last_tool_result
        self._init_reconnection_callbacks(on_reconnecting, on_reconnected)
        # Flag: when True, _handle_evt_response_done will send response.create
        self._pending_response_create = False
        self._enable_manual_rehydrate_response_input = enable_manual_rehydrate_response_input
        self._rehydration_input_prefix = list(rehydration_input_prefix or [])
        self._awaiting_manual_audio_commit = False
        self._manual_committed_audio_item_id: Optional[str] = None
        self._pending_manual_tool_results: list[dict[str, str]] = []
        self._awaiting_manual_tool_continuation_start = False
        self._manual_tool_continuation_retry_count = 0
        self._manual_tool_continuation_retry_task: Optional[asyncio.Task] = None
        self._manual_response_in_flight = False
        self._last_manual_commit_monotonic = 0.0

    def _manual_rehydrate_mode_active(self) -> bool:
        return self._enable_manual_rehydrate_response_input

    async def _handle_context(self, context):
        """Bypass Pipecat auto-response/context replay in manual rehydrate mode.

        In manual rehydrate mode we trigger generation via explicit
        ``response.create(input=...)`` after audio commit and after tool completion.
        If we let the base ``_handle_context`` run, a later context update from a tool
        result causes Pipecat to treat it as the first context setup and issue its own
        ``_create_response()``, which races with our manual continuation path.
        """
        if not self._manual_rehydrate_mode_active():
            await super()._handle_context(context)
            return

        self._context = context
        # Prevent accidental fallback _create_response() calls from replaying the entire
        # local context into the remote conversation during manual response.create(input).
        self._llm_needs_conversation_setup = False

        # Keep local completed-tool bookkeeping in sync without sending tool results or
        # auto-triggering another response; manual mode does both explicitly.
        await super()._process_completed_function_calls(send_new_results=False)

    async def _process_completed_function_calls(self, send_new_results: bool):
        """Disable base auto-send/auto-response for tool results in manual mode."""
        if self._manual_rehydrate_mode_active():
            return
        await super()._process_completed_function_calls(send_new_results)

    def _build_manual_response_input(
        self, *, include_rehydration_prefix_and_audio: bool
    ) -> list[dict[str, Any]]:
        # Initial response should include rehydrated history + the committed user audio.
        # Tool continuations should not replay that audio/history, otherwise the model can
        # re-interpret the same turn and re-issue duplicate tool calls.
        response_input: list[dict[str, Any]] = []
        if include_rehydration_prefix_and_audio:
            response_input.extend(self._rehydration_input_prefix)
            if self._manual_committed_audio_item_id:
                response_input.append({"type": "item_reference", "id": self._manual_committed_audio_item_id})
        for item in self._pending_manual_tool_results:
            response_input.append(
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
            response_input.append(
                {
                    "type": "function_call_output",
                    "status": "completed",
                    "call_id": item["call_id"],
                    "output": item["output"],
                }
            )
        return response_input

    async def _send_response_create_with_optional_input(
        self,
        *,
        include_rehydration_input: bool,
        include_rehydration_prefix_and_audio: bool = True,
    ):
        """Send response.create, optionally with explicit `response.input`."""
        if include_rehydration_input and not self._manual_rehydrate_mode_active():
            include_rehydration_input = False

        payload: dict[str, Any] = {
            "type": "response.create",
            "response": {
                "output_modalities": self._get_enabled_modalities(),
            },
        }
        if include_rehydration_input:
            payload["response"]["input"] = self._build_manual_response_input(
                include_rehydration_prefix_and_audio=include_rehydration_prefix_and_audio
            )
            self._manual_response_in_flight = True

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        await self._ws_send(payload)
        suffix = " with response.input" if include_rehydration_input else ""
        logger.info(f"[OpenAI Realtime] Sent response.create{suffix}")

    def _clear_manual_tool_continuation_state(self):
        self._awaiting_manual_tool_continuation_start = False
        self._manual_tool_continuation_retry_count = 0
        self._pending_manual_tool_results = []
        if self._manual_tool_continuation_retry_task and not self._manual_tool_continuation_retry_task.done():
            self._manual_tool_continuation_retry_task.cancel()
        self._manual_tool_continuation_retry_task = None

    async def _retry_manual_tool_continuation_after_delay(self):
        delay = min(1.0, 0.25 * max(1, self._manual_tool_continuation_retry_count))
        logger.warning(
            f"[OpenAI Realtime] Retrying manual tool continuation response.create in {delay:.2f}s "
            f"(attempt {self._manual_tool_continuation_retry_count})"
        )
        try:
            await asyncio.sleep(delay)
            if not self._awaiting_manual_tool_continuation_start or not self._pending_manual_tool_results:
                return
            await self._send_response_create_with_optional_input(
                include_rehydration_input=True,
                include_rehydration_prefix_and_audio=False,
            )
        finally:
            self._manual_tool_continuation_retry_task = None

    async def _handle_user_stopped_speaking(self, frame):
        """Manual response.create path for rehydrate mode when server VAD is disabled."""
        if not self._manual_rehydrate_mode_active():
            await super()._handle_user_stopped_speaking(frame)
            return

        turn_detection_disabled = (
            self._session_properties.audio
            and self._session_properties.audio.input
            and self._session_properties.audio.input.turn_detection is False
        )
        if not turn_detection_disabled:
            logger.warning(
                "[OpenAI Realtime] Manual rehydrate response-input mode requested but turn_detection "
                "is not disabled; falling back to base handling"
            )
            await super()._handle_user_stopped_speaking(frame)
            return

        if self._awaiting_manual_audio_commit:
            logger.warning("[OpenAI Realtime] Already waiting for input_audio_buffer.committed; ignoring duplicate stop event")
            return

        if self._manual_response_in_flight:
            logger.debug("[OpenAI Realtime] Ignoring user stop event while response is still in flight")
            return

        now = time.monotonic()
        if now - self._last_manual_commit_monotonic < 0.75:
            logger.debug("[OpenAI Realtime] Debouncing duplicate user stop event")
            return

        self._awaiting_manual_audio_commit = True
        await self.send_client_event(rt_events.InputAudioBufferCommitEvent())
        self._last_manual_commit_monotonic = now
        logger.info("[OpenAI Realtime] Sent input_audio_buffer.commit (awaiting committed item_id)")

    async def _handle_evt_input_audio_buffer_committed(self, evt):
        if not self._manual_rehydrate_mode_active():
            return
        if not self._awaiting_manual_audio_commit:
            logger.debug(
                f"[OpenAI Realtime] input_audio_buffer.committed received without pending manual commit (item_id={evt.item_id})"
            )
            return

        self._awaiting_manual_audio_commit = False
        self._manual_committed_audio_item_id = evt.item_id
        self._clear_manual_tool_continuation_state()
        logger.info(
            f"[OpenAI Realtime] Input audio committed (item_id={evt.item_id}); sending manual response.create with rehydration input"
        )
        await self._send_response_create_with_optional_input(
            include_rehydration_input=True,
            include_rehydration_prefix_and_audio=True,
        )

    async def _handle_evt_audio_delta(self, evt):
        if self._awaiting_manual_tool_continuation_start:
            self._clear_manual_tool_continuation_state()
        await super()._handle_evt_audio_delta(evt)

    async def _handle_evt_text_delta(self, evt):
        if self._awaiting_manual_tool_continuation_start:
            self._clear_manual_tool_continuation_state()
        await super()._handle_evt_text_delta(evt)

    async def _handle_evt_audio_transcript_delta(self, evt):
        if self._awaiting_manual_tool_continuation_start:
            self._clear_manual_tool_continuation_state()
        await super()._handle_evt_audio_transcript_delta(evt)

    async def _handle_evt_function_call_arguments_done(self, evt):
        """Handle function call completion: run tool, send result, defer response.create.

        We completely override (no super()) to prevent the base class's
        _process_completed_function_calls path from sending response.create
        during the active response.
        """
        call_id = getattr(evt, "call_id", None)
        if call_id is None:
            return

        # --- Replicate the base class's tool execution (without its auto-send) ---
        try:
            args = json.loads(evt.arguments)
            function_call_item = self._pending_function_calls.get(call_id)
            if function_call_item:
                del self._pending_function_calls[call_id]

                # Pre-mark this tool_call_id as completed BEFORE run_function_calls
                # so _process_completed_function_calls() won't auto-send it
                self._completed_tool_calls.add(call_id)

                function_calls = [
                    FunctionCallFromLLM(
                        context=self._context,
                        tool_call_id=call_id,
                        function_name=function_call_item.name,
                        arguments=args,
                    )
                ]
                await self.run_function_calls(function_calls)
                logger.debug(f"[OpenAI Realtime] Processed function call: {function_call_item.name}")
            else:
                logger.warning(f"[OpenAI Realtime] No tracked function call for call_id: {call_id}")
                return
        except Exception as e:
            logger.error(f"[OpenAI Realtime] Failed to process function call: {e}")
            return

        # --- Send our explicit tool result ---
        tool_result = (
            self._get_last_tool_result()
            if self._get_last_tool_result
            else {"status": "success"}
        )
        output_json = json.dumps(tool_result)
        if self._manual_rehydrate_mode_active():
            self._pending_manual_tool_results.append(
                {
                    "call_id": call_id,
                    "name": function_call_item.name,
                    "arguments": evt.arguments,
                    "output": output_json,
                }
            )
            logger.info(
                f"[OpenAI Realtime] Queued tool result for manual response.create input "
                f"(call_id={call_id})"
            )
        else:
            tool_output_item_id = uuid.uuid4().hex
            create_ev = rt_events.ConversationItemCreateEvent(
                item=rt_events.ConversationItem(
                    id=tool_output_item_id,
                    type="function_call_output",
                    call_id=call_id,
                    output=output_json,
                )
            )
            await self.send_client_event(create_ev)
            logger.info(f"[OpenAI Realtime] Sent function_call_output for call_id={call_id}")

        # Defer response.create until the current response is complete (response.done).
        self._pending_response_create = True
        logger.info("[OpenAI Realtime] Deferred response.create until response.done")

    async def _handle_evt_response_done(self, evt):
        """Handle response.done: call super, then send deferred response.create if needed.

        This fires after the active response is complete, so it is safe to
        send response.create to trigger the model to continue with the tool
        result in context.
        """
        await super()._handle_evt_response_done(evt)
        self._manual_response_in_flight = False

        if self._pending_response_create:
            self._pending_response_create = False
            if self._manual_rehydrate_mode_active() and self._pending_manual_tool_results:
                await asyncio.sleep(0.25)
                self._awaiting_manual_tool_continuation_start = True
                self._manual_tool_continuation_retry_count = 1
                await self._send_response_create_with_optional_input(
                    include_rehydration_input=True,
                    include_rehydration_prefix_and_audio=False,
                )
            else:
                await self.send_client_event(rt_events.ResponseCreateEvent())
                logger.info("[OpenAI Realtime] Sent deferred response.create after response.done")

    async def _handle_evt_error(self, evt):
        if (
            self._manual_rehydrate_mode_active()
            and self._awaiting_manual_tool_continuation_start
            and getattr(evt, "error", None) is not None
            and evt.error.code == "conversation_already_has_active_response"
            and self._pending_manual_tool_results
        ):
            if self._manual_tool_continuation_retry_count < 3:
                self._manual_tool_continuation_retry_count += 1
                if (
                    self._manual_tool_continuation_retry_task is None
                    or self._manual_tool_continuation_retry_task.done()
                ):
                    self._manual_tool_continuation_retry_task = self.create_task(
                        self._retry_manual_tool_continuation_after_delay()
                    )
                return True
        if (
            self._manual_rehydrate_mode_active()
            and getattr(evt, "error", None) is not None
            and evt.error.code == "conversation_already_has_active_response"
        ):
            # Another response is already running; don't treat this as fatal.
            self._awaiting_manual_audio_commit = False
            self._manual_response_in_flight = True
            logger.warning(
                "[OpenAI Realtime] Ignoring conversation_already_has_active_response in manual mode; "
                "waiting for current response to finish"
            )
            return True
        await super()._handle_evt_error(evt)
        return False

    async def _receive_task_handler(self):
        """Use base receive loop unless manual rehydrate mode needs committed-audio events."""
        if not self._manual_rehydrate_mode_active():
            await super()._receive_task_handler()
            return

        async for message in self._websocket:
            evt = rt_events.parse_server_event(message)
            if evt.type == "session.created":
                await self._handle_evt_session_created(evt)
            elif evt.type == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif evt.type == "response.output_audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.output_audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type == "conversation.item.added":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "conversation.item.done":
                await self._handle_evt_conversation_item_done(evt)
            elif evt.type == "conversation.item.input_audio_transcription.delta":
                await self._handle_evt_input_audio_transcription_delta(evt)
            elif evt.type == "conversation.item.input_audio_transcription.completed":
                await self.handle_evt_input_audio_transcription_completed(evt)
            elif evt.type == "conversation.item.retrieved":
                await self._handle_conversation_item_retrieved(evt)
            elif evt.type == "response.done":
                await self._handle_evt_response_done(evt)
            elif evt.type == "input_audio_buffer.speech_started":
                await self._handle_evt_speech_started(evt)
            elif evt.type == "input_audio_buffer.speech_stopped":
                await self._handle_evt_speech_stopped(evt)
            elif evt.type == "input_audio_buffer.committed":
                await self._handle_evt_input_audio_buffer_committed(evt)
            elif evt.type == "response.output_text.delta":
                await self._handle_evt_text_delta(evt)
            elif evt.type == "response.output_audio_transcript.delta":
                await self._handle_evt_audio_transcript_delta(evt)
            elif evt.type == "response.function_call_arguments.done":
                await self._handle_evt_function_call_arguments_done(evt)
            elif evt.type == "error":
                if not await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                    handled = await self._handle_evt_error(evt)
                    if not handled:
                        return
        await self._handle_ws_close()
