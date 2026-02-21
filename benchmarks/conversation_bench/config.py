"""Configuration for the conversation bench benchmark."""
from pathlib import Path

from .turns import turns
from .tools import ToolsSchemaForTest
from .system import system_instruction


class BenchmarkConfig:
    """Configuration for the conversation bench benchmark."""

    name = "conversation_bench"
    description = "75-turn hard benchmark with ~12K token knowledge base and 9 tools"
    hf_repo = "arcada-labs/conversation-bench"

    turns = turns
    tools_schema = ToolsSchemaForTest
    system_instruction = system_instruction

    _benchmark_dir = Path(__file__).parent
    audio_dir = _benchmark_dir / "audio"

    @classmethod
    def get_audio_path(cls, turn_index: int) -> Path:
        """Get the audio file path for a specific turn."""
        if not cls.audio_dir.exists() or not any(cls.audio_dir.glob("*.wav")):
            raise FileNotFoundError(
                f"No audio files found in {cls.audio_dir}. "
                f"Generate them first: uv run python scripts/generate_audio.py {cls.name}"
            )
        return cls.audio_dir / f"turn_{turn_index:03d}.wav"
