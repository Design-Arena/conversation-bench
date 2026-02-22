"""Audio Arena: Multi-turn voice AI evaluation framework CLI.

Usage:
    uv run audio-arena run grocery_bench --model gpt-realtime
    uv run audio-arena run grocery_bench --model nova-sonic --rehydrate
    uv run audio-arena judge runs/grocery_bench/20251213T123456_gpt-realtime
    uv run audio-arena list-benchmarks
"""

import asyncio
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Service Aliases
# ============================================================================

SERVICE_ALIASES = {
    "openai": "pipecat.services.openai.llm.OpenAILLMService",
    "openai-realtime": "audio_arena.pipelines.openai_realtime.OpenAIRealtimeLLMServiceExplicitToolResult",
    "openrouter": "pipecat.services.openai.llm.OpenAILLMService",  # OpenRouter uses OpenAI-compatible API
    "anthropic": "pipecat.services.anthropic.llm.AnthropicLLMService",
    "google": "pipecat.services.google.llm.GoogleLLMService",
    "gemini-live": "audio_arena.pipelines.realtime.GeminiLiveLLMServiceWithReconnection",
    "bedrock": "pipecat.services.aws.llm.AWSBedrockLLMService",
    "groq": "pipecat.services.groq.llm.GroqLLMService",
    "cerebras": "pipecat.services.cerebras.llm.CerebrasLLMService",
    "ultravox-realtime": "pipecat.services.ultravox.llm.UltravoxRealtimeLLMService",
}


# ============================================================================
# Model Aliases — friendly names → (actual_api_model, default_service, default_pipeline)
# ============================================================================

MODEL_ALIASES: dict[str, tuple[str, Optional[str], Optional[str]]] = {
    "gpt-realtime":      ("gpt-realtime",                     "openai-realtime",   None),
    "gemini-native-audio": ("gemini-2.5-flash-native-audio",  "gemini-live",       None),
    "ultravox":          ("ultravox-v0.7",                     "ultravox-realtime", None),
    "grok-realtime":     ("grok-realtime",                     None,               None),
    "nova-sonic":        ("amazon.nova-2-sonic-v1:0",          None,               "nova-sonic"),
}


# ============================================================================
# Pipeline Registry
# ============================================================================

PIPELINE_CLASSES = {
    "text": "audio_arena.pipelines.text.TextPipeline",
    "realtime": "audio_arena.pipelines.realtime.RealtimePipeline",
    "grok-realtime": "audio_arena.pipelines.grok_realtime.GrokRealtimePipeline",
    "nova-sonic": "audio_arena.pipelines.nova_sonic.NovaSonicPipeline",
}


# ============================================================================
# Utility Functions
# ============================================================================


def resolve_model_alias(
    model: str,
    service: Optional[str] = None,
    pipeline: Optional[str] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve a model alias to (actual_model, service, pipeline).

    If the model is a known alias, fills in service/pipeline defaults
    (only when not explicitly provided by the caller).
    """
    if model in MODEL_ALIASES:
        actual_model, default_service, default_pipeline = MODEL_ALIASES[model]
        return (
            actual_model,
            service or default_service,
            pipeline or default_pipeline,
        )
    return model, service, pipeline


def load_service_class(service: str) -> type:
    """Load service class from fully qualified name or alias."""
    class_name = SERVICE_ALIASES.get(service, service)
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def load_benchmark(name: str):
    """Load benchmark by name from benchmarks/ directory."""
    try:
        module = importlib.import_module(f"benchmarks.{name}.config")
        return module.BenchmarkConfig
    except ModuleNotFoundError as e:
        raise click.UsageError(f"Benchmark '{name}' not found: {e}")


def list_available_benchmarks() -> list[str]:
    """Discover available benchmarks by scanning benchmarks/ directory."""
    # Find the benchmarks directory relative to the package or current working directory
    cwd_benchmarks = Path.cwd() / "benchmarks"

    benchmarks = []
    if cwd_benchmarks.exists():
        for d in cwd_benchmarks.iterdir():
            if d.is_dir() and not d.name.startswith("_") and (d / "config.py").exists():
                benchmarks.append(d.name)

    return sorted(benchmarks)


def get_pipeline_class(pipeline_type: str) -> type:
    """Load pipeline class by type name."""
    class_name = PIPELINE_CLASSES.get(pipeline_type)
    if not class_name:
        raise click.UsageError(
            f"Unknown pipeline: {pipeline_type}. Available: {list(PIPELINE_CLASSES.keys())}"
        )
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def infer_pipeline(model: str) -> str:
    """Infer default pipeline from model name pattern."""
    m = model.lower()
    # Grok realtime uses dedicated pipeline for xAI-specific protocol handling
    if m.startswith("grok") and "realtime" in m:
        return "grok-realtime"
    if "realtime" in m:
        return "realtime"
    if "native-audio" in m or "live" in m:
        return "realtime"
    if "ultravox" in m:
        return "realtime"
    if "nova-sonic" in m or "nova_sonic" in m:
        return "nova-sonic"
    return "text"


def create_run_directory(benchmark_name: str, model: str) -> Path:
    """Create timestamped run directory."""
    import uuid

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    # Add a short unique suffix to prevent collisions in parallel runs
    unique_suffix = str(uuid.uuid4())[:8]
    # Sanitize model name for filesystem (replace / and :)
    safe_model = model.replace("/", "_").replace(":", "_")
    run_dir = (
        Path("runs") / benchmark_name / f"{timestamp}_{safe_model}_{unique_suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool = False):
    """Configure logging to both console and run directory."""
    level = logging.DEBUG if verbose else logging.INFO

    # Remove default loguru handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level="INFO" if not verbose else "DEBUG",
        format="<level>{message}</level>",
    )

    # File handler (always DEBUG for debugging failed runs)
    logger.add(
        run_dir / "run.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {name}: {message}",
    )


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
def cli():
    """Audio Arena: Multi-turn voice AI evaluation framework."""
    pass


@cli.command()
@click.argument("benchmark_name")
@click.option("--model", required=True, help="Model name (e.g., gpt-4o, claude-sonnet-4-5)")
@click.option("--service", help="Service class name or alias (e.g., openai, anthropic)")
@click.option(
    "--pipeline",
    help="Pipeline type (text, realtime, nova-sonic). Auto-detected if not specified.",
)
@click.option("--only-turns", help="Comma-separated turn indices to run (e.g., 0,1,2)")
@click.option(
    "--rehydrate",
    is_flag=True,
    help="Single-step rehydration mode: evaluate each turn independently with golden prior context.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline: Optional[str],
    only_turns: Optional[str],
    rehydrate: bool,
    verbose: bool,
):
    """Run a benchmark against an LLM.

    Model aliases (auto-fill service & pipeline):
        gpt-realtime, gemini-native-audio, ultravox, grok-realtime, nova-sonic

    Examples:
        uv run audio-arena run conversation_bench --model gpt-realtime
        uv run audio-arena run conversation_bench --model nova-sonic
        uv run audio-arena run conversation_bench --model gemini-native-audio --rehydrate
        uv run audio-arena run conversation_bench --model claude-sonnet-4-5 --service anthropic
    """
    model, service, pipeline = resolve_model_alias(model, service, pipeline)

    if rehydrate:
        asyncio.run(
            _run_rehydrated(benchmark_name, model, service, pipeline, only_turns, verbose)
        )
    else:
        asyncio.run(_run(benchmark_name, model, service, pipeline, only_turns, verbose))


async def _run(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline_type: Optional[str],
    only_turns: Optional[str],
    verbose: bool,
):
    """Async implementation of the run command."""
    # Load benchmark
    BenchmarkConfig = load_benchmark(benchmark_name)
    benchmark = BenchmarkConfig()

    # Infer pipeline if not specified
    if not pipeline_type:
        pipeline_type = infer_pipeline(model)
        click.echo(f"Auto-detected pipeline: {pipeline_type}")

    pipeline_cls = get_pipeline_class(pipeline_type)

    # Check if pipeline requires a service
    requires_service = getattr(pipeline_cls, "requires_service", True)
    if requires_service and not service:
        raise click.UsageError(f"--service is required for {pipeline_type} pipeline")

    # Load service class if provided
    service_class = load_service_class(service) if service else None

    # Create output directory
    run_dir = create_run_directory(benchmark_name, model)
    click.echo(f"Output directory: {run_dir}")

    # Setup logging
    setup_logging(run_dir, verbose)

    # Create recorder
    from audio_arena.recording.transcript_recorder import TranscriptRecorder

    recorder = TranscriptRecorder(run_dir, model)

    # Parse turn indices if provided
    turn_indices = None
    if only_turns:
        turn_indices = [int(i.strip()) for i in only_turns.split(",")]
        click.echo(f"Running only turns: {turn_indices}")

    # Run the pipeline
    try:
        pipeline_instance = pipeline_cls(benchmark)
        await pipeline_instance.run(
            recorder=recorder,
            model=model,
            service_class=service_class,
            service_name=service,
            turn_indices=turn_indices,
        )
        click.echo(f"Completed benchmark run")
        click.echo(f"  Transcript: {run_dir / 'transcript.jsonl'}")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise click.ClickException(str(e))
    finally:
        recorder.close()


async def _run_rehydrated(
    benchmark_name: str,
    model: str,
    service: Optional[str],
    pipeline_type: Optional[str],
    only_turns: Optional[str],
    verbose: bool,
):
    """Run benchmark in single-step rehydration mode.

    Each turn is evaluated independently: the model receives golden conversation
    history for all prior turns, then live audio/text for the target turn only.
    A fresh API session is created per turn to ensure complete isolation.
    """
    from audio_arena.recording.transcript_recorder import TranscriptRecorder

    BenchmarkConfig = load_benchmark(benchmark_name)
    benchmark = BenchmarkConfig()
    all_turns = benchmark.turns

    if not pipeline_type:
        pipeline_type = infer_pipeline(model)
        click.echo(f"Auto-detected pipeline: {pipeline_type}")

    pipeline_cls = get_pipeline_class(pipeline_type)

    requires_service = getattr(pipeline_cls, "requires_service", True)
    if requires_service and not service:
        raise click.UsageError(f"--service is required for {pipeline_type} pipeline")

    service_class = load_service_class(service) if service else None

    run_dir = create_run_directory(benchmark_name, model)
    click.echo(f"Output directory: {run_dir}")
    click.echo(f"Mode: single-step rehydration ({len(all_turns)} total turns)")

    setup_logging(run_dir, verbose)

    target_indices = list(range(len(all_turns)))
    if only_turns:
        target_indices = [int(i.strip()) for i in only_turns.split(",")]
        click.echo(f"Evaluating turns: {target_indices}")

    succeeded = 0
    failed_turns = []

    for target_idx in target_indices:
        golden_history = all_turns[:target_idx] if target_idx > 0 else None
        click.echo(
            f"[Rehydration] Turn {target_idx}/{len(all_turns) - 1}"
            + (f" (rehydrating {len(golden_history)} golden turns)" if golden_history else "")
        )

        recorder = TranscriptRecorder(run_dir, model)
        pipeline_instance = pipeline_cls(benchmark)

        try:
            await pipeline_instance.run(
                recorder=recorder,
                model=model,
                service_class=service_class,
                service_name=service,
                turn_indices=[target_idx],
                rehydration_turns=golden_history,
            )
            succeeded += 1
        except Exception as e:
            logger.exception(f"Turn {target_idx} failed: {e}")
            click.echo(f"  Turn {target_idx} FAILED: {e}")
            failed_turns.append(target_idx)
        finally:
            recorder.close()

    # Write final runtime summary covering all turns
    runtime = {
        "model_name": model,
        "turns": succeeded,
        "total_attempted": len(target_indices),
        "failed_turns": failed_turns,
        "mode": "rehydrated",
        "note": "Single-step rehydration: each turn evaluated independently with golden prior context",
    }
    (run_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )

    click.echo(f"\nCompleted rehydrated run: {succeeded}/{len(target_indices)} turns succeeded")
    if failed_turns:
        click.echo(f"  Failed turns: {failed_turns}")
    click.echo(f"  Transcript: {run_dir / 'transcript.jsonl'}")


NON_CONVO_BENCHMARKS = {"appointment_bench", "event_bench", "grocery_bench"}


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--only-turns", help="Comma-separated turn indices to judge (e.g., 0,1,2)")
@click.option(
    "--judge",
    "judge_backend",
    type=click.Choice(["claude", "openai"], case_sensitive=False),
    default=None,
    help="Judge backend to use. Defaults to 'openai' for non-convo benchmarks, 'claude' for conversation_bench.",
)
@click.option("--judge-model", default=None, help="Model for judging (default: claude-opus-4-5 or o3)")
@click.option("--skip-turn-taking", is_flag=True, help="Skip audio turn-taking analysis (faster; all turns count as turn_taking=True)")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def judge(
    run_dir: str,
    only_turns: Optional[str],
    judge_backend: Optional[str],
    judge_model: Optional[str],
    skip_turn_taking: bool,
    debug: bool,
):
    """Judge a completed benchmark run.

    Examples:
        uv run audio-arena judge runs/grocery_bench/20251213T123456_gpt-4o
        uv run audio-arena judge runs/conversation_bench/... --judge claude
        uv run audio-arena judge runs/appointment_bench/... --judge openai --judge-model gpt-4.1
    """
    run_path = Path(run_dir)

    # Infer benchmark from path: runs/{benchmark}/{timestamp}_{model}/
    benchmark_name = run_path.parent.name

    # Auto-select judge backend based on benchmark type
    if judge_backend is None:
        if benchmark_name in NON_CONVO_BENCHMARKS:
            judge_backend = "openai"
        else:
            judge_backend = "claude"
    click.echo(f"Using {judge_backend} judge for {benchmark_name}")

    # Load transcript
    transcript_path = run_path / "transcript.jsonl"
    if not transcript_path.exists():
        raise click.UsageError(f"No transcript found at {transcript_path}")

    # Parse turn indices
    turn_indices_set: Optional[set[int]] = None
    if only_turns:
        turn_indices_set = {int(i.strip()) for i in only_turns.split(",")}

    # Load benchmark for expected turns and get_relevant_dimensions
    get_relevant_dimensions_fn = None
    try:
        BenchmarkConfig = load_benchmark(benchmark_name)
        benchmark = BenchmarkConfig()
        expected_turns = benchmark.turns
        benchmark_turns_module = importlib.import_module(f"benchmarks.{benchmark_name}.turns")
        get_relevant_dimensions_fn = getattr(benchmark_turns_module, 'get_relevant_dimensions', None)
    except Exception:
        click.echo(f"Could not load benchmark '{benchmark_name}', using shared turns module")
        from benchmarks.conversation_bench.turns import turns as expected_turns

    # Load shared utilities
    from audio_arena.judging.llm_judge import load_transcript, write_outputs

    records = load_transcript(run_path)
    if turn_indices_set is not None:
        records = [r for r in records if r["turn"] in turn_indices_set]

    if judge_backend == "openai":
        from audio_arena.judging.openai_judge import judge_with_openai, OPENAI_JUDGE_VERSION, OPENAI_JUDGE_MODEL

        effective_model = judge_model or OPENAI_JUDGE_MODEL
        try:
            result = asyncio.run(
                judge_with_openai(
                    run_path,
                    only_turns=turn_indices_set,
                    debug=debug,
                    expected_turns=expected_turns,
                    skip_turn_taking=skip_turn_taking,
                    get_relevant_dimensions_fn=get_relevant_dimensions_fn,
                    model=judge_model,
                )
            )
        except Exception as e:
            raise click.ClickException(f"Judgment failed: {e}")

        write_outputs(
            run_path,
            records,
            result["judgments"],
            result["summary"],
            result["model_name"],
            result.get("realignment_notes", ""),
            result.get("function_tracking", {}),
            result.get("turn_taking_analysis"),
            expected_turns=expected_turns,
            judge_name="openai",
            judge_version=OPENAI_JUDGE_VERSION,
            judge_model=result.get("judge_model", effective_model),
        )
        summary_file = "openai_summary.json"

    else:
        from audio_arena.judging.llm_judge import judge_with_claude

        try:
            result = asyncio.run(
                judge_with_claude(
                    run_path,
                    only_turns=turn_indices_set,
                    debug=debug,
                    expected_turns=expected_turns,
                    skip_turn_taking=skip_turn_taking,
                    get_relevant_dimensions_fn=get_relevant_dimensions_fn,
                )
            )
        except Exception as e:
            raise click.ClickException(f"Judgment failed: {e}")

        write_outputs(
            run_path,
            records,
            result["judgments"],
            result["summary"],
            result["model_name"],
            result.get("realignment_notes", ""),
            result.get("function_tracking", {}),
            result.get("turn_taking_analysis"),
            expected_turns=expected_turns,
            judge_name="claude",
        )
        summary_file = "claude_summary.json"

    # Print summary
    summary_path = run_path / summary_file
    summary = json.loads(summary_path.read_text())
    passes = summary.get("passes", summary.get("claude_passes", {}))
    total = summary.get("turns_scored", 0)

    click.echo(f"\nJudged {total} turns (with turn-taking analysis)")
    click.echo(f"  Turn-taking: {passes.get('turn_taking', total)}/{total}")
    click.echo(f"  Tool use: {passes.get('tool_use_correct', 0)}/{total}")
    click.echo(f"  Instruction following: {passes.get('instruction_following', 0)}/{total}")
    click.echo(f"  KB grounding: {passes.get('kb_grounding', 0)}/{total}")

    category_totals = summary.get("category_totals", {})
    amb_total = category_totals.get("ambiguity_handling", 0)
    state_total = category_totals.get("state_tracking", 0)
    if amb_total:
        click.echo(f"  Ambiguity handling: {passes.get('ambiguity_handling', 0)}/{amb_total}")
    if state_total:
        click.echo(f"  State tracking: {passes.get('state_tracking', 0)}/{state_total}")

    turn_taking_failures = summary.get("turn_taking_failures", [])
    if turn_taking_failures:
        click.echo(f"\nTurn-taking failures: {turn_taking_failures}")


@cli.command("list-benchmarks")
def list_benchmarks():
    """List available benchmarks."""
    benchmarks = list_available_benchmarks()
    if not benchmarks:
        click.echo("No benchmarks found in benchmarks/ directory")
        return

    click.echo("Available benchmarks:")
    for name in benchmarks:
        try:
            BenchmarkConfig = load_benchmark(name)
            description = getattr(BenchmarkConfig, "description", "")
            click.echo(f"  {name}: {description}")
        except Exception:
            click.echo(f"  {name}")


@cli.command("list-pipelines")
def list_pipelines():
    """List available pipelines."""
    click.echo("Available pipelines:")
    for name, cls_path in PIPELINE_CLASSES.items():
        click.echo(f"  {name}: {cls_path}")


@cli.command("list-aliases")
def list_aliases():
    """List service aliases."""
    click.echo("Service aliases:")
    for alias, cls_path in SERVICE_ALIASES.items():
        click.echo(f"  {alias}: {cls_path}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
