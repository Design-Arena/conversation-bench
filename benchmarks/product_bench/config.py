"""Configuration for the product comparison benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the product comparison benchmark."""

    name = "product_bench"
    description = "31-turn laptop comparison benchmark: multi-intent turns, retroactive correction via reported speech, vague pronoun resolution, conditional arithmetic chains, discount stacking policy edge, 3 subtle + 2 direct false memory traps, cross-reference counting, 3-step order modification chain, out-of-scope deflection"
    hf_repo = "arcada-labs/product-bench"

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
