"""Configuration for the personal assistant conversation benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the personal assistant conversation benchmark."""

    name = "assistant_bench"
    description = ("31-turn personal assistant benchmark: dual requests in single turns, "
                   "topic switching mid-conversation, late references to early topics, "
                   "intent segmentation across flight booking / email / calendar / reminders, "
                   "mid-sentence self-correction, retroactive email correction, "
                   "total-cost cross-reference calculation, correction-chain recall, "
                   "vague pronoun disambiguation, email count cross-reference, "
                   "ambiguous entity disambiguation + over-clarification traps, "
                   "combined recap+modify, 3 false memory traps + correction-chain trap, "
                   "audio traps on name spelling / airport codes / dates / times")
    hf_repo = "arcada-labs/assistant-bench"

    turns = turns
    tools_schema = ToolsSchemaForTest
    system_instruction = system_instruction

    _benchmark_dir = Path(__file__).parent
    audio_dir = _benchmark_dir / "audio"

    @classmethod
    def get_audio_path(cls, turn_index: int) -> Path:
        """Get the audio file path for a specific turn, downloading from HF if needed."""
        if not cls.audio_dir.exists() or not any(cls.audio_dir.glob("*.wav")):
            from audio_arena.data import ensure_audio
            cls.audio_dir = ensure_audio(cls._benchmark_dir, cls.hf_repo)
        return cls.audio_dir / f"turn_{turn_index:03d}.wav"
