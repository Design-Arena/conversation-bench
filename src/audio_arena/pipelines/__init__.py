"""Pipeline implementations for different LLM service types.

Pipelines handle the full execution of multi-turn benchmarks including:
- Creating and configuring LLM services
- Managing turn flow (queuing turns, detecting end-of-turn)
- Recording transcripts and metrics
- Handling reconnection for long-running sessions

Available pipelines:
- TextPipeline: For text-based LLM services (OpenAI, Anthropic, Google, etc.)
- RealtimePipeline: For speech-to-speech services (OpenAI Realtime, Gemini Live)
- GrokRealtimePipeline: For xAI Grok Voice Agent API
- NovaSonicPipeline: For AWS Nova Sonic speech-to-speech service
"""

from audio_arena.pipelines.base import BasePipeline
from audio_arena.pipelines.text import TextPipeline
from audio_arena.pipelines.realtime import (
    RealtimePipeline,
    GeminiLiveLLMServiceWithReconnection,
)
from audio_arena.pipelines.grok_realtime import (
    GrokRealtimePipeline,
    XAIRealtimeLLMService,
)
from audio_arena.pipelines.nova_sonic import (
    NovaSonicPipeline,
    NovaSonicLLMServiceWithCompletionSignal,
    NovaSonicTurnGate,
)

__all__ = [
    "BasePipeline",
    "TextPipeline",
    "RealtimePipeline",
    "GeminiLiveLLMServiceWithReconnection",
    "GrokRealtimePipeline",
    "XAIRealtimeLLMService",
    "NovaSonicPipeline",
    "NovaSonicLLMServiceWithCompletionSignal",
    "NovaSonicTurnGate",
]
