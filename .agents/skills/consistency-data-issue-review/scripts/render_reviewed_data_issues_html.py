import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a final HTML report from manually reviewed candidate data-issue rows."
        )
    )
    parser.add_argument(
        "--reviewed-csv",
        required=True,
        help=(
            "CSV produced from candidate_data_issues.csv after LLM/manual review. "
            "Expected columns include benchmark_name, turn, manual_review_status, "
            "final_issue_label, final_issue_summary, evidence_summary, reviewer_notes."
        ),
    )
    parser.add_argument(
        "--output-html",
        required=True,
        help="Destination HTML file.",
    )
    parser.add_argument(
        "--model-label",
        default="model",
        help="Display label for the reviewed model/run set.",
    )
    return parser.parse_args()


def normalize_status(value: object) -> str:
    return str(value or "").strip().lower()


def render_html(output_path: Path, model_label: str, reviewed_df: pd.DataFrame) -> None:
    confirmed_df = reviewed_df[reviewed_df["manual_review_status"] == "confirmed_issue"].copy()
    possible_df = reviewed_df[reviewed_df["manual_review_status"] == "possible_issue"].copy()
    rejected_df = reviewed_df[reviewed_df["manual_review_status"] == "rejected"].copy()

    def render_cards(frame: pd.DataFrame, badge_class: str, badge_label: str) -> str:
        if frame.empty:
            return '<article class="card empty"><p>No rows in this category.</p></article>'

        cards: list[str] = []
        for row in frame.itertuples():
            cards.append(
                f"""
                <article class="card">
                  <div class="card-header">
                    <h3>{row.benchmark_name}</h3>
                    <span class="tag {badge_class}">{badge_label}</span>
                  </div>
                  <div class="turn">Turn <code>{row.turn}</code></div>
                  <p class="why"><strong>{row.final_issue_label or 'Reviewed issue'}</strong></p>
                  <p class="detail"><strong>Candidate types:</strong> <code>{row.candidate_issue_types or 'none'}</code></p>
                  <p class="detail"><strong>Pattern:</strong> <code>{row.overall_status_pattern}</code></p>
                  <p class="detail"><strong>Metric flips:</strong> <code>{row.metric_flips or 'none'}</code></p>
                  <p class="detail"><strong>User turn:</strong> {row.user_text}</p>
                  <div class="reason"><strong>Final summary:</strong> {row.final_issue_summary or row.reason_summary}</div>
                  <div class="reason"><strong>Evidence:</strong> {row.evidence_summary or row.reason_summary}</div>
                  <div class="reason"><strong>Reviewer notes:</strong> {row.reviewer_notes or 'None provided.'}</div>
                </article>
                """
            )
        return "\n".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reviewed Data Issues Report</title>
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
      --neutral: #36566e;
      --neutral-soft: #d9e7f1;
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
    .tag-confirmed {{ background: var(--warn-soft); color: var(--warn); }}
    .tag-possible {{ background: var(--caution-soft); color: var(--caution); }}
    .tag-rejected {{ background: var(--neutral-soft); color: var(--neutral); }}
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
      <div class="eyebrow">Manual Review</div>
      <h1>Reviewed Data / Judge Issues</h1>
      <p>
        This final report is built only from datapoints that were selected after LLM/manual review
        of the candidate packet for <strong>{model_label}</strong>.
      </p>
    </section>

    <section class="section">
      <h2>Confirmed Issues</h2>
      <div class="grid">
        {render_cards(confirmed_df, "tag-confirmed", "Confirmed")}
      </div>
    </section>

    <section class="section">
      <h2>Possible Issues</h2>
      <div class="grid">
        {render_cards(possible_df, "tag-possible", "Possible")}
      </div>
    </section>

    <section class="section">
      <h2>Rejected Candidates</h2>
      <div class="grid">
        {render_cards(rejected_df, "tag-rejected", "Rejected")}
      </div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    reviewed_csv_path = Path(args.reviewed_csv).expanduser().resolve()
    output_html_path = Path(args.output_html).expanduser().resolve()

    reviewed_df = pd.read_csv(reviewed_csv_path).copy()
    if reviewed_df.empty:
        raise ValueError(f"No rows found in {reviewed_csv_path}")

    required_columns = [
        "benchmark_name",
        "turn",
        "manual_review_status",
        "candidate_issue_types",
        "overall_status_pattern",
        "metric_flips",
        "user_text",
        "reason_summary",
    ]
    missing_columns = [column for column in required_columns if column not in reviewed_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in reviewed CSV: {missing_columns}")

    reviewed_df["manual_review_status"] = reviewed_df["manual_review_status"].map(normalize_status)
    render_html(output_html_path, args.model_label, reviewed_df)
    print(output_html_path)


if __name__ == "__main__":
    main()
