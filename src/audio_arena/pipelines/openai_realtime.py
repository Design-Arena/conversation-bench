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

import json
from typing import Callable, Optional

from loguru import logger

from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class OpenAIRealtimeLLMServiceExplicitToolResult(OpenAIRealtimeLLMService):
    """OpenAI Realtime service that explicitly sends tool results to the API.

    Completely takes over function call handling to avoid the
    "conversation_already_has_active_response" error that occurs when
    response.create is sent during an active response.

    Flow:
    1. response.function_call_arguments.done fires (during active response)
    2. We run the tool handler ourselves (run_function_calls)
    3. We pre-mark the call as completed to prevent base class auto-send
    4. We send conversation.item.create (function_call_output) with our result
    5. response.done fires (response is now complete)
    6. We send response.create to trigger the model to continue
    """

    def __init__(self, get_last_tool_result: Optional[Callable[[], dict]] = None, **kwargs):
        """See module docstring 'Why the constructor takes get_last_tool_result'.
        When provided, we call it when sending function_call_output; else we send {"status": "success"}.
        """
        super().__init__(**kwargs)
        self._get_last_tool_result = get_last_tool_result
        # Flag: when True, _handle_evt_response_done will send response.create
        self._pending_response_create = False

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
        create_ev = rt_events.ConversationItemCreateEvent(
            item=rt_events.ConversationItem(
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

        if self._pending_response_create:
            self._pending_response_create = False
            await self.send_client_event(rt_events.ResponseCreateEvent())
            logger.info("[OpenAI Realtime] Sent deferred response.create after response.done")
