import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
REPO_SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(REPO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS_DIR))

from analyze_consistency_runs import load_comparison_details, summarize_benchmarks, summarize_metric_flips, summarize_turns  # noqa: E402


DEFAULT_BENCHMARKS = [
    "appointment_bench",
    "conversation_bench",
    "event_bench",
    "grocery_bench",
]

LIKELY_DATA_ISSUE_TYPES = {
    "judge_inconsistency": "Judge inconsistency",
    "gold_overspecification": "Gold overspecification",
    "tool_match_overspecification": "Tool-match overspecification",
    "tool_first_false_pass": "Tool-first false pass",
}


@dataclass
class ComparisonRun:
    benchmark: str
    comparison_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated consistency studies or analyze existing comparison folders, "
            "then produce suspected data/judge issue reports and HTML."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        action="append",
        default=[],
        help="Existing compare_model_runs output directory. Repeat for multiple benchmarks.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="Benchmarks to run when comparison dirs are not provided.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for fresh repeated runs.",
    )
    parser.add_argument(
        "--runs-per-model",
        type=int,
        default=3,
        help="Number of repeated runs for fresh comparisons.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Parallelism for fresh rehydrated runs.",
    )
    parser.add_argument(
        "--service",
        default="openai-realtime",
        help="Service override for fresh runs.",
    )
    parser.add_argument(
        "--judge",
        default="openai",
        help="Judge backend for fresh runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Combined output directory for the review artifacts.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> str:
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"stdout tail:\n{process.stdout[-4000:]}\n"
            f"stderr tail:\n{process.stderr[-4000:]}"
        )
    return process.stdout


def run_fresh_comparisons(args: argparse.Namespace, timestamp: str) -> list[ComparisonRun]:
    if not args.model:
        raise ValueError("--model is required when no --comparison-dir values are supplied.")

    comparison_runs: list[ComparisonRun] = []
    for benchmark in args.benchmarks:
        comparison_dir = (
            REPO_ROOT
            / "results"
            / benchmark
            / f"{timestamp}_{args.model}_consistency_{args.runs_per_model}runs"
        )
        command = [
            "uv",
            "run",
            "python",
            str(REPO_ROOT / "scripts" / "compare_model_runs.py"),
            "--benchmark",
            benchmark,
            "--models",
            args.model,
            "--runs-per-model",
            str(args.runs_per_model),
            "--service",
            args.service,
            "--parallel",
            str(args.parallel),
            "--judge",
            args.judge,
            "--output-dir",
            str(comparison_dir),
        ]
        run_command(command)
        comparison_runs.append(ComparisonRun(benchmark=benchmark, comparison_dir=comparison_dir))
    return comparison_runs


def load_existing_comparisons(comparison_dirs: list[str]) -> list[ComparisonRun]:
    comparison_runs: list[ComparisonRun] = []
    for comparison_dir_value in comparison_dirs:
        comparison_dir = Path(comparison_dir_value).expanduser().resolve()
        metadata_path = comparison_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {comparison_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparison_runs.append(
            ComparisonRun(
                benchmark=str(metadata.get("benchmark") or comparison_dir.parent.name),
                comparison_dir=comparison_dir,
            )
        )
    return comparison_runs


def run_combined_consistency_analysis(
    comparison_runs: list[ComparisonRun],
    output_dir: Path,
) -> None:
    command = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "analyze_consistency_runs.py"),
        *[str(run.comparison_dir) for run in comparison_runs],
        "--output-dir",
        str(output_dir),
    ]
    run_command(command)


def classify_issue_type(row: pd.Series) -> tuple[str | None, str | None]:
    reasoning = str(row.get("judge_reasoning", "")).lower()
    assistant_text = str(row.get("assistant_text", "")).lower()
    tool_calls_json = str(row.get("tool_calls_json", "")).lower()
    overall_status = str(row.get("overall_status", "")).lower()
    failing_metrics = str(row.get("failing_metrics", "")).lower()

    if (
        "same" in reasoning and "different" in reasoning
    ):
        return "judge_inconsistency", "Judge rationale itself signals inconsistent equivalence handling."

    if (
        "semantically equivalent" in reasoning and "false" in reasoning
    ) or (
        "added constraints" in reasoning and "tool_use_correct is false" in reasoning
    ):
        return "tool_match_overspecification", "Compatible tool-call variation is treated as wrong."

    if (
        "omitted the room detail" in reasoning
        or "golden also states" in reasoning
        or "incomplete vs golden" in reasoning
        or "the golden response also clarifies" in reasoning
    ):
        return "gold_overspecification", "The row is failed for missing gold-only detail rather than missing the user's requested answer."

    if (
        overall_status == "pass"
        and tool_calls_json not in {"", "[]"}
        and (
            "couldn't" in assistant_text
            or "wasn't able" in assistant_text
            or "i don't have a way" in assistant_text
            or "not seeing" in assistant_text and "tool_use_correct=true" in reasoning
        )
    ):
        return "tool_first_false_pass", "The required tool call happened, but the assistant text contradicts the action."

    if (
        "tool_use_correct is true" in reasoning
        and (
            "contradicts itself" in reasoning
            or "words-actions mismatch" in reasoning
            or "conflicts with the tool action" in reasoning
        )
        and overall_status == "pass"
    ):
        return "tool_first_false_pass", "The grader passes the row despite acknowledging contradictory assistant text."

    if (
        "case difference" in reasoning
        and "semantically equivalent" in reasoning
        and "ambiguity_handling" in failing_metrics
    ):
        return "judge_inconsistency", "The rationale accepts the tool equivalence but still appears overly sensitive elsewhere."

    return None, None


def build_candidate_issue_rows(comparison_runs: list[ComparisonRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    review_rows: list[dict] = []

    for comparison_run in comparison_runs:
        details_df, _ = load_comparison_details(comparison_run.comparison_dir)
        turn_summary_df = summarize_turns(details_df)

        unstable_turns = turn_summary_df[
            turn_summary_df["unstable_overall"] | turn_summary_df["unstable_metric"]
        ][["benchmark_name", "turn", "overall_status_pattern", "metric_flips"]]

        unstable_details_df = details_df.merge(
            unstable_turns,
            on=["benchmark_name", "turn"],
            how="inner",
        )

        grouped = unstable_details_df.groupby(["benchmark_name", "turn"], sort=True)
        for (benchmark_name, turn), turn_df in grouped:
            ordered_df = turn_df.sort_values("run_index")
            statuses = ordered_df["overall_status"].tolist()
            unique_statuses = set(statuses)
            issue_labels: set[str] = set()
            reasons: list[str] = []

            for _, row in ordered_df.iterrows():
                issue_type, reason = classify_issue_type(row)
                if issue_type:
                    issue_labels.add(issue_type)
                    if reason and reason not in reasons:
                        reasons.append(reason)

            if not issue_labels:
                continue

            confidence = "possible"
            if "judge_inconsistency" in issue_labels or "tool_first_false_pass" in issue_labels:
                confidence = "strong"

            summary_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "turn": int(turn),
                    "confidence": confidence,
                    "candidate_issue_types": ",".join(sorted(issue_labels)),
                    "overall_status_pattern": ordered_df["overall_status_pattern"].iloc[0],
                    "metric_flips": ordered_df["metric_flips"].iloc[0],
                    "user_text": ordered_df["user_text"].iloc[0],
                    "reason_summary": " ".join(reasons).strip(),
                    "run_count": int(len(ordered_df)),
                    "pass_runs": int((ordered_df["overall_status"] == "pass").sum()),
                    "fail_runs": int((ordered_df["overall_status"] == "fail").sum()),
                    "manual_review_status": "needs_review",
                    "final_issue_label": "",
                    "final_issue_summary": "",
                    "evidence_summary": "",
                    "reviewer_notes": "",
                }
            )

            for _, row in ordered_df.iterrows():
                review_rows.append(
                    {
                        "benchmark_name": benchmark_name,
                        "turn": int(turn),
                        "confidence": confidence,
                        "candidate_issue_types": ",".join(sorted(issue_labels)),
                        "manual_review_status": "needs_review",
                        "run_index": int(row["run_index"]),
                        "run_id": row["run_id"],
                        "overall_status": row["overall_status"],
                        "failing_metrics": row["failing_metrics"],
                        "metric_flips": ordered_df["metric_flips"].iloc[0],
                        "overall_status_pattern": ordered_df["overall_status_pattern"].iloc[0],
                        "user_text": row["user_text"],
                        "assistant_text": row["assistant_text"],
                        "tool_calls_json": row["tool_calls_json"],
                        "judge_reasoning": row["judge_reasoning"],
                        "reason_summary": " ".join(reasons).strip(),
                        "reviewer_notes": "",
                    }
                )

    suspected_df = pd.DataFrame(summary_rows)
    review_df = pd.DataFrame(review_rows)
    if suspected_df.empty:
        return suspected_df, review_df

    confidence_order = {"strong": 0, "possible": 1}
    suspected_df["confidence_rank"] = suspected_df["confidence"].map(confidence_order).fillna(9)
    suspected_df = suspected_df.sort_values(
        ["confidence_rank", "benchmark_name", "turn"],
        ascending=[True, True, True],
    ).drop(columns=["confidence_rank"])
    review_df = review_df.sort_values(
        ["benchmark_name", "turn", "run_index"],
        ascending=[True, True, True],
    )
    return suspected_df.reset_index(drop=True), review_df.reset_index(drop=True)


def write_markdown_report(
    *,
    output_path: Path,
    suspected_issues_df: pd.DataFrame,
    benchmark_summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# Candidate Data / Judge Issues",
        "",
        "These are heuristic candidates only.",
        "Manual review is required before treating any row here as a confirmed benchmark or judge issue.",
        "",
        "## Benchmark Summary",
    ]

    for row in benchmark_summary_df.itertuples():
        lines.append(
            f"- {row.benchmark_name}: unstable_overall_turns={row.unstable_overall_turns}/{row.turns}, "
            f"unstable_metric_turns={row.unstable_metric_turns}/{row.turns}"
        )

    strong_df = suspected_issues_df[suspected_issues_df["confidence"] == "strong"]
    possible_df = suspected_issues_df[suspected_issues_df["confidence"] != "strong"]

    for title, frame in [("Strong Candidates", strong_df), ("Possible Candidates", possible_df)]:
        lines.extend(["", f"## {title}"])
        if frame.empty:
            lines.append("- None")
            continue
        for row in frame.itertuples():
            lines.append(
                f"- {row.benchmark_name} turn {row.turn}: {row.candidate_issue_types} | "
                f"overall={row.overall_status_pattern} | metrics={row.metric_flips or 'none'}"
            )
            lines.append(f"  - User: {row.user_text}")
            lines.append(f"  - Why: {row.reason_summary}")
            lines.append(f"  - Manual review: {row.manual_review_status}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_html(
    *,
    output_path: Path,
    model_label: str,
    suspected_issues_df: pd.DataFrame,
) -> None:
    strong_df = suspected_issues_df[suspected_issues_df["confidence"] == "strong"]
    possible_df = suspected_issues_df[suspected_issues_df["confidence"] != "strong"]

    def render_cards(frame: pd.DataFrame, badge_class: str, badge_label: str) -> str:
        if frame.empty:
            return '<article class="card empty"><p>No rows in this category.</p></article>'

        cards: list[str] = []
        for row in frame.itertuples():
            issue_labels = [
                LIKELY_DATA_ISSUE_TYPES.get(issue_type, issue_type)
                for issue_type in str(row.candidate_issue_types).split(",")
                if issue_type
            ]
            issue_text = ", ".join(issue_labels)
            cards.append(
                f"""
                <article class="card">
                  <div class="card-header">
                    <h3>{row.benchmark_name}</h3>
                    <span class="tag {badge_class}">{badge_label}</span>
                  </div>
                  <div class="turn">Turn <code>{row.turn}</code></div>
                  <p class="why">{issue_text}</p>
                  <p class="detail"><strong>Pattern:</strong> <code>{row.overall_status_pattern}</code></p>
                  <p class="detail"><strong>Metric flips:</strong> <code>{row.metric_flips or 'none'}</code></p>
                  <p class="detail"><strong>User turn:</strong> {row.user_text}</p>
                  <p class="detail"><strong>Manual review:</strong> <code>{row.manual_review_status}</code></p>
                  <div class="reason"><strong>Why suspicious:</strong> {row.reason_summary}</div>
                </article>
                """
            )
        return "\n".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Candidate Data Issues Review</title>
  <style>
    :root {{
      --paper: #f6f1e8;
      --panel: #fffaf2;
      --ink: #2c2823;
      --muted: #6f675d;
      --line: #d9d0c3;
      --accent: #c05a1a;
      --warn: #8d2f2f;
      --warn-soft: #f4dada;
      --caution: #8b6a16;
      --caution-soft: #f3ebc9;
      --shadow: rgba(44, 40, 35, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", Avenir, "Helvetica Neue", Helvetica, Arial, "DejaVu Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, #f2e1cf 0, transparent 28%),
        radial-gradient(circle at left 20%, #efe6d8 0, transparent 32%),
        linear-gradient(180deg, #fbf7f1 0%, var(--paper) 100%);
      line-height: 1.5;
    }}
    .page {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px 64px; }}
    .hero {{
      padding: 32px;
      border: 1px solid var(--line);
      background: rgba(255, 250, 242, 0.86);
      box-shadow: 0 18px 40px var(--shadow);
      border-radius: 24px;
      margin-bottom: 28px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 10px;
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 600; letter-spacing: -0.02em; }}
    h1 {{ font-size: clamp(32px, 5vw, 52px); line-height: 1.02; margin-bottom: 14px; }}
    h2 {{ font-size: 26px; margin-bottom: 16px; }}
    h3 {{ font-size: 18px; margin-bottom: 10px; }}
    p {{ margin: 0 0 12px; color: var(--muted); }}
    .section {{ margin-top: 32px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px 18px 16px;
      box-shadow: 0 10px 24px var(--shadow);
    }}
    .card.empty {{ display: flex; align-items: center; justify-content: center; min-height: 120px; }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .tag {{
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .tag-strong {{ background: var(--warn-soft); color: var(--warn); }}
    .tag-possible {{ background: var(--caution-soft); color: var(--caution); }}
    .turn {{ font-size: 14px; color: var(--muted); margin-bottom: 10px; }}
    .why {{ color: var(--ink); margin-bottom: 10px; }}
    .detail {{ font-size: 14px; }}
    .reason {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px dashed var(--line);
      color: var(--ink);
      font-size: 14px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
      background: #f3ece3;
      padding: 2px 6px;
      border-radius: 6px;
      color: #3a3129;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="eyebrow">Consistency Review</div>
      <h1>Candidate Data / Judge Issues</h1>
      <p>
        This report highlights unstable datapoints from repeated runs of <strong>{model_label}</strong>
        that look more like benchmark-spec, gold-answer, or judge-behavior problems than clean model failures.
        These are review candidates only. Manual inspection is still required.
      </p>
    </section>

    <section class="section">
      <h2>Strong Candidates</h2>
      <div class="grid">
        {render_cards(strong_df, "tag-strong", "Strong")}
      </div>
    </section>

    <section class="section">
      <h2>Possible Candidates</h2>
      <div class="grid">
        {render_cards(possible_df, "tag-possible", "Possible")}
      </div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    if args.comparison_dir:
        comparison_runs = load_existing_comparisons(args.comparison_dir)
        model_label = args.model or "model"
    else:
        comparison_runs = run_fresh_comparisons(args, timestamp)
        model_label = str(args.model)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = REPO_ROOT / "results" / f"consistency_{timestamp}_{model_label}_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_combined_consistency_analysis(comparison_runs, output_dir)

    details_frames = []
    for comparison_run in comparison_runs:
        details_df, _ = load_comparison_details(comparison_run.comparison_dir)
        details_frames.append(details_df)
    combined_details_df = pd.concat(details_frames, ignore_index=True)
    turn_summary_df = summarize_turns(combined_details_df)
    metric_flip_df = summarize_metric_flips(combined_details_df)
    benchmark_summary_df = summarize_benchmarks(turn_summary_df)
    suspected_issues_df, review_packet_df = build_candidate_issue_rows(comparison_runs)

    suspected_csv_path = output_dir / "candidate_data_issues.csv"
    suspected_md_path = output_dir / "candidate_data_issues.md"
    suspected_html_path = output_dir / "candidate_data_issues.html"
    review_packet_csv_path = output_dir / "candidate_review_packet.csv"
    review_packet_jsonl_path = output_dir / "candidate_review_packet.jsonl"
    benchmark_summary_path = output_dir / "benchmark_variance_summary.csv"
    metric_flip_summary_path = output_dir / "metric_flip_summary.csv"

    suspected_issues_df.to_csv(suspected_csv_path, index=False, encoding="utf-8")
    review_packet_df.to_csv(review_packet_csv_path, index=False, encoding="utf-8")
    with review_packet_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in review_packet_df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    benchmark_summary_df.to_csv(benchmark_summary_path, index=False, encoding="utf-8")
    metric_flip_df["metric"].value_counts().rename_axis("metric").reset_index(name="flip_count").to_csv(
        metric_flip_summary_path,
        index=False,
        encoding="utf-8",
    )

    write_markdown_report(
        output_path=suspected_md_path,
        suspected_issues_df=suspected_issues_df,
        benchmark_summary_df=benchmark_summary_df,
    )
    render_html(
        output_path=suspected_html_path,
        model_label=model_label,
        suspected_issues_df=suspected_issues_df,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "comparison_dirs": [str(run.comparison_dir) for run in comparison_runs],
                "benchmark_summary_csv": str(benchmark_summary_path),
                "candidate_data_issues_csv": str(suspected_csv_path),
                "candidate_data_issues_md": str(suspected_md_path),
                "candidate_data_issues_html": str(suspected_html_path),
                "candidate_review_packet_csv": str(review_packet_csv_path),
                "candidate_review_packet_jsonl": str(review_packet_jsonl_path),
                "metric_flip_summary_csv": str(metric_flip_summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
