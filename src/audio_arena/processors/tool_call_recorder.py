from typing import Any, Callable, Optional, Set

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ToolCallRecorder(FrameProcessor):
    """Records tool calls and results into the TranscriptRecorder.

    This processor is transport-agnostic and relies on LLMService emitting
    FunctionCallInProgressFrame and FunctionCallResultFrame while executing
    tool calls. We append minimal details to the active `recorder`.

    Duplicate tool calls (detected by the pipeline's _function_catchall) are:
    - RECORDED in the transcript (for debugging/analysis)
    - NOT PUSHED downstream (to prevent context pollution)

    This ensures the judge can see duplicates and score appropriately,
    while keeping the LLM context clean for future turns.
    """

    def __init__(
        self,
        recorder_ref: Callable[[], Any],
        duplicate_ids_ref: Optional[Callable[[], Set[str]]] = None,
    ):
        super().__init__()
        # `recorder_ref` is a zero-arg callable returning the current recorder
        self._recorder_ref = recorder_ref
        # `duplicate_ids_ref` returns the set of tool_call_ids that are duplicates
        self._duplicate_ids_ref = duplicate_ids_ref

    def _rec(self):
        try:
            return self._recorder_ref() if callable(self._recorder_ref) else None
        except Exception:
            return None

    def _get_duplicate_ids(self) -> Set[str]:
        """Get the current set of duplicate tool_call_ids."""
        if self._duplicate_ids_ref and callable(self._duplicate_ids_ref):
            try:
                return self._duplicate_ids_ref()
            except Exception:
                return set()
        return set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only process DOWNSTREAM frames to avoid duplication
        # (LLM service emits frames in both directions)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            tool_call_id = getattr(frame, 'tool_call_id', None)
            is_duplicate = tool_call_id in self._get_duplicate_ids() if tool_call_id else False

            rec = self._rec()
            if rec is not None:
                try:
                    # ALWAYS record the tool call (including duplicates) for transcript
                    rec.record_tool_call(
                        frame.function_name,
                        frame.arguments or {},
                        is_duplicate=is_duplicate,
                    )
                    log_prefix = "[DUPLICATE] " if is_duplicate else ""
                    logger.info(
                        f"[TOOL_RECORDER] {log_prefix}FunctionCallInProgressFrame "
                        f"name={frame.function_name} args={frame.arguments} "
                        f"tool_call_id={tool_call_id}"
                    )
                except Exception as e:
                    logger.debug(f"ToolCallRecorder: failed to record call: {e}")

            # Only push non-duplicates downstream to prevent context pollution
            if not is_duplicate:
                await self.push_frame(frame, direction)
            else:
                logger.debug(
                    f"ToolCallRecorder: NOT pushing duplicate FunctionCallInProgressFrame "
                    f"to context (tool_call_id={tool_call_id})"
                )

        elif isinstance(frame, FunctionCallResultFrame):
            tool_call_id = frame.tool_call_id
            is_duplicate = tool_call_id in self._get_duplicate_ids() if tool_call_id else False

            rec = self._rec()
            if rec is not None:
                try:
                    # ALWAYS record the result (including duplicates) for transcript
                    rec.record_tool_result(
                        frame.function_name,
                        {
                            "tool_call_id": tool_call_id,
                            "result": frame.result,
                            "properties": getattr(frame, "properties", None),
                            "is_duplicate": is_duplicate,
                        },
                    )
                    log_prefix = "[DUPLICATE] " if is_duplicate else ""
                    logger.info(
                        f"[TOOL_RECORDER] {log_prefix}FunctionCallResultFrame "
                        f"name={frame.function_name} tool_call_id={tool_call_id} "
                        f"result_keys={list((frame.result or {}).keys())}"
                    )
                except Exception as e:
                    logger.debug(f"ToolCallRecorder: failed to record result: {e}")

            # Only push non-duplicates downstream to prevent context pollution
            if not is_duplicate:
                await self.push_frame(frame, direction)
            else:
                logger.debug(
                    f"ToolCallRecorder: NOT pushing duplicate FunctionCallResultFrame "
                    f"to context (tool_call_id={tool_call_id})"
                )

        else:
            # All other frames pass through unchanged
            await self.push_frame(frame, direction)
