import asyncio
import json
import sys
import types
from pathlib import Path

from audio_arena import cli


def test_get_disable_vad_status_messages():
    assert cli.get_disable_vad_status_messages(
        disable_vad=False,
        rehydrate=False,
        pipeline_type="realtime",
        service="openai-realtime",
        model="gpt-realtime",
    ) == []

    assert cli.get_disable_vad_status_messages(
        disable_vad=True,
        rehydrate=False,
        pipeline_type="text",
        service="openai-realtime",
        model="gpt-realtime",
    ) == ["[disable-vad] Ignored: --disable-vad only applies to the realtime pipeline."]

    assert cli.get_disable_vad_status_messages(
        disable_vad=True,
        rehydrate=False,
        pipeline_type="realtime",
        service="gemini-live",
        model="gpt-realtime",
    ) == [
        "[disable-vad] Ignored: supported only for --service openai-realtime (got: gemini-live).",
    ]

    assert cli.get_disable_vad_status_messages(
        disable_vad=True,
        rehydrate=False,
        pipeline_type="realtime",
        service="openai-realtime",
        model="gemini-2.5-flash-native-audio-preview-12-2025",
    ) == [
        "[disable-vad] Ignored: model 'gemini-2.5-flash-native-audio-preview-12-2025' is not an OpenAI realtime model.",
    ]

    assert cli.get_disable_vad_status_messages(
        disable_vad=True,
        rehydrate=False,
        pipeline_type="realtime",
        service="openai-realtime",
        model="gpt-realtime",
    ) == [
        "[disable-vad] Active: server-side VAD disabled for OpenAI Realtime.",
        "[disable-vad] Note: manual response.create(input=...) hydration only applies with --rehydrate.",
    ]

    assert cli.get_disable_vad_status_messages(
        disable_vad=True,
        rehydrate=True,
        pipeline_type="realtime",
        service="openai-realtime",
        model="gpt-realtime",
    ) == [
        "[disable-vad] Active: server-side VAD disabled for OpenAI Realtime.",
        "[disable-vad] Active: using manual response.create(input=...) rehydration flow.",
    ]


def test_run_rehydrated_parallel_executes_concurrently(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    class FakeBenchmarkConfig:
        name = "fake_bench"

        def __init__(self):
            self.turns = [
                {"input": "turn-0"},
                {"input": "turn-1"},
                {"input": "turn-2"},
                {"input": "turn-3"},
            ]

    class FakePipeline:
        requires_service = True
        active_runs = 0
        max_parallel = 0
        calls: list[dict] = []

        def __init__(self, benchmark):
            self.benchmark = benchmark

        async def run(
            self,
            *,
            recorder,
            model,
            service_class=None,
            service_name=None,
            turn_indices=None,
            rehydration_turns=None,
            disable_vad=False,
        ):
            type(self).active_runs += 1
            type(self).max_parallel = max(type(self).max_parallel, type(self).active_runs)
            type(self).calls.append(
                {
                    "model": model,
                    "service_name": service_name,
                    "turn_indices": turn_indices,
                    "rehydration_turns": rehydration_turns,
                    "disable_vad": disable_vad,
                    "service_class": service_class,
                }
            )
            await asyncio.sleep(0.05)
            type(self).active_runs -= 1

    class FakeRecorder:
        def __init__(self, run_dir: Path, model_name: str):
            self.run_dir = run_dir
            self.model_name = model_name

        def close(self):
            return None

    fake_recorder_module = types.SimpleNamespace(TranscriptRecorder=FakeRecorder)
    monkeypatch.setitem(sys.modules, "audio_arena.recording.transcript_recorder", fake_recorder_module)
    monkeypatch.setattr(cli, "load_benchmark", lambda name: FakeBenchmarkConfig)
    monkeypatch.setattr(cli, "get_pipeline_class", lambda pipeline_type: FakePipeline)
    monkeypatch.setattr(cli, "create_run_directory", lambda benchmark_name, model: run_dir)
    monkeypatch.setattr(cli, "setup_logging", lambda run_dir, verbose: None)
    monkeypatch.setattr(cli, "load_service_class", lambda service: object())

    echoed: list[str] = []
    monkeypatch.setattr(cli.click, "echo", lambda message: echoed.append(str(message)))

    asyncio.run(
        cli._run_rehydrated(
            benchmark_name="fake_bench",
            model="gpt-realtime",
            service="openai-realtime",
            pipeline_type="realtime",
            only_turns=None,
            verbose=False,
            max_parallel=3,
            disable_vad=True,
        )
    )

    assert FakePipeline.max_parallel >= 2
    assert len(FakePipeline.calls) == 4
    assert all(call["disable_vad"] is True for call in FakePipeline.calls)
    assert all(call["service_name"] == "openai-realtime" for call in FakePipeline.calls)

    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["mode"] == "rehydrated"
    assert runtime["parallel"] == 3
    assert runtime["disable_vad"] is True
    assert runtime["turns"] == 4

    assert any("manual response.create(input=...) rehydration flow" in line for line in echoed)
