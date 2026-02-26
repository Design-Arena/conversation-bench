import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_model_runs import (
    aggregate_metric_summary,
    aggregate_run_summary,
    build_run_record,
    create_count_plot,
    create_markdown_summary,
    create_metric_plot,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compare_model_runs-style outputs from an existing set of judged run "
            "directories."
        )
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name for the selected run directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for run_results.csv, plots, and metadata.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more judged run directories to include in the comparison output.",
    )
    parser.add_argument(
        "--judge",
        default="openai",
        help="Judge backend used for these runs.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional judge model override for metadata only.",
    )
    parser.add_argument(
        "--service",
        default="openai-realtime",
        help="Service override for metadata only.",
    )
    parser.add_argument(
        "--skip-turn-taking",
        action="store_true",
        default=True,
        help="Whether these runs were judged with skip-turn-taking enabled.",
    )
    return parser.parse_args()


def infer_model_name(run_dir: Path) -> str:
    parts = run_dir.name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected run directory name: {run_dir.name}")
    return "_".join(parts[1:-1])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(run_dir).expanduser().resolve() for run_dir in args.run_dirs]
    for run_dir in run_dirs:
        for required_name in ["runtime.json", "openai_summary.json", "openai_judged.jsonl"]:
            required_path = run_dir / required_name
            if not required_path.exists():
                raise FileNotFoundError(f"Missing {required_name} in {run_dir}")

    run_records: list[dict] = []
    command_records: list[dict] = []

    for run_index, run_dir in enumerate(run_dirs, start=1):
        model_name = infer_model_name(run_dir)
        summary_payload = json.loads((run_dir / "openai_summary.json").read_text(encoding="utf-8"))
        runtime_payload = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
        judged_rows = load_jsonl(run_dir / "openai_judged.jsonl")
        run_records.append(
            build_run_record(
                model_name=model_name,
                run_index=run_index,
                run_dir=run_dir,
                summary_payload=summary_payload,
                runtime_payload=runtime_payload,
                judged_rows=judged_rows,
            )
        )
        command_records.extend(
            [
                {
                    "phase": "run",
                    "model_name": model_name,
                    "run_index": run_index,
                    "command": "",
                    "run_dir": str(run_dir),
                },
                {
                    "phase": "judge",
                    "model_name": model_name,
                    "run_index": run_index,
                    "command": "",
                    "run_dir": str(run_dir),
                },
            ]
        )

    run_results_df = pd.DataFrame(run_records)
    if run_results_df.empty:
        raise ValueError("No judged runs were provided.")

    metric_summary_df = aggregate_metric_summary(run_results_df)
    summary_df = aggregate_run_summary(run_results_df)

    run_results_df.to_csv(output_dir / "run_results.csv", index=False, encoding="utf-8")
    metric_summary_df.to_csv(output_dir / "metric_summary.csv", index=False, encoding="utf-8")
    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(command_records).to_csv(output_dir / "commands.csv", index=False, encoding="utf-8")

    create_metric_plot(metric_summary_df, output_dir / "pass_rates_by_metric.png")
    create_count_plot(
        summary_df,
        mean_column="fail_rows_mean",
        lower_column="fail_rows_ci_lower_mean",
        upper_column="fail_rows_ci_upper_mean",
        title="Failed Turns by Model",
        y_label="Failed turns",
        output_path=output_dir / "failed_turns.png",
    )
    create_count_plot(
        summary_df,
        mean_column="latency_ms_p95_mean",
        lower_column="latency_ms_p95_ci_lower_mean",
        upper_column="latency_ms_p95_ci_upper_mean",
        title="Latency p95 by Model",
        y_label="Latency p95 (ms)",
        output_path=output_dir / "latency_p95.png",
    )
    create_count_plot(
        summary_df,
        mean_column="model_ended_session_count_mean",
        lower_column="model_ended_session_count_ci_lower_mean",
        upper_column="model_ended_session_count_ci_upper_mean",
        title="MODEL_ENDED_SESSION Count by Model",
        y_label="Turns",
        output_path=output_dir / "model_ended_session.png",
    )

    models = run_results_df["model_name"].drop_duplicates().tolist()
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "benchmark": args.benchmark,
        "models": models,
        "runs_per_model": len(run_dirs),
        "service_override": args.service,
        "rehydrate": True,
        "disable_vad": bool(run_results_df["disable_vad"].dropna().any()),
        "parallel": int(pd.to_numeric(run_results_df["runtime_parallel"], errors="coerce").dropna().max()),
        "judge": args.judge,
        "judge_model": args.judge_model,
        "skip_turn_taking": args.skip_turn_taking,
        "output_dir": str(output_dir),
    }
    create_markdown_summary(
        argparse.Namespace(
            benchmark=args.benchmark,
            service=args.service,
            rehydrate=metadata["rehydrate"],
            disable_vad=metadata["disable_vad"],
            parallel=metadata["parallel"],
            judge=args.judge,
            judge_model=args.judge_model,
            skip_turn_taking=args.skip_turn_taking,
            runs_per_model=len(run_dirs),
        ),
        run_results_df,
        metric_summary_df,
        summary_df,
        output_dir / "summary.md",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
