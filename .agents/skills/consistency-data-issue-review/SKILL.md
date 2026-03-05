---
name: consistency-data-issue-review
description: Run repeated benchmark consistency studies, find turn-level pass/fail flips across runs of the same model, identify suspected data-spec or grader issues versus real model failures, and generate a polished HTML review. Use when asked to run N repeated evals, inspect unstable datapoints, separate likely benchmark/judge problems from model problems, or produce a consistency review report.
---

# Consistency Data Issue Review

Use this skill for repeated-run consistency analysis on judged benchmark outputs.

## When to use it

- The user wants `N` repeated runs for one model across one or more benchmarks.
- The goal is to find rows that flip between pass and fail.
- The user wants to know whether those flips look like:
  - real model instability
  - grader inconsistency
  - benchmark / gold-answer overspecification
  - false passes caused by tool-first grading
- The user wants a presentable HTML report.

## Source of truth

- Use the run-indexed `consistency_*` artifacts as the primary source of truth for repeated runs of the same model.
- Do **not** use the existing deep-error-analysis helper as the main repeated-run source, because it keys by `model_name` and can collapse same-model runs together.
- For model behavior and grading details, rely on:
  - `openai_judged.jsonl`
  - `consistency_run_details.csv`
  - `consistency_turn_summary.csv`
  - `consistency_metric_flips.csv`
  - `consistency_flip_details.csv`

## Preferred entrypoint

Run:

```bash
uv run python .agents/skills/consistency-data-issue-review/scripts/review_consistency_data_issues.py \
  --model gpt-realtime-1.5 \
  --runs-per-model 3 \
  --benchmarks appointment_bench conversation_bench event_bench grocery_bench
```

Or, for existing comparison folders:

```bash
uv run python .agents/skills/consistency-data-issue-review/scripts/review_consistency_data_issues.py \
  --comparison-dir /abs/path/to/results/appointment_bench/..._consistency_3runs \
  --comparison-dir /abs/path/to/results/conversation_bench/..._consistency_3runs
```

After the LLM manually reviews the candidate CSV and fills in the review columns, render the final HTML with:

```bash
uv run python .agents/skills/consistency-data-issue-review/scripts/render_reviewed_data_issues_html.py \
  --reviewed-csv /abs/path/to/candidate_data_issues.csv \
  --output-html /abs/path/to/reviewed_data_issues.html \
  --model-label gpt-realtime-1.5
```

## What the script does

1. If no `--comparison-dir` is provided, it runs `scripts/compare_model_runs.py` once per benchmark.
2. It runs `scripts/analyze_consistency_runs.py` on the resulting comparison folders.
3. It filters suspicious unstable rows using heuristics for:
   - `judge_inconsistency`
   - `gold_overspecification`
   - `tool_first_false_pass`
   - `tool_match_overspecification`
4. It writes:
   - combined consistency artifacts
   - `candidate_data_issues.csv`
   - `candidate_data_issues.md`
   - `candidate_data_issues.html`
   - a row-level review packet for manual inspection
5. After the LLM/manual review is complete, use `render_reviewed_data_issues_html.py` to create the final HTML from the reviewed selections only.

## Required manual review step

The script is only a candidate generator. It is **not** allowed to be the final judge of whether a datapoint is bad.

After the heuristic filter, manually review the candidate rows in depth:

1. Read the run-by-run assistant outputs for that turn.
2. Read the full `judge_reasoning` for each run.
3. Read the `tool_calls` and `tool_results` where relevant.
4. Compare the passing and failing variants directly.
5. Decide whether the row is:
   - likely real model instability
   - likely grader inconsistency
   - likely benchmark / gold overspecification
   - likely false pass / false fail

Do not present the candidate list as final findings until this manual pass is complete.

The HTML produced by the script should therefore be treated as:
- a review queue
- a shortlist for manual inspection

Not as the final answer by itself.

## Review rules

- Only call something a data/judge issue when the evidence is stronger than “the model failed.”
- Favor `judge_inconsistency` when near-equivalent behaviors receive different labels.
- Favor `gold_overspecification` when the response answers the user’s real question but fails for omitted gold-only detail.
- Favor `tool_match_overspecification` when the tool call is broader but still semantically on target.
- Favor `tool_first_false_pass` when the required tool call happens but the assistant text clearly says the opposite or otherwise fails the user request.
- When the heuristic suggests an issue, verify it by manually comparing at least two concrete runs before making the final call.
- If the row still looks ambiguous after manual review, keep it in a `possible` or `needs-manual-review` bucket instead of upgrading it to a definitive issue.
- Keep real model failures out of the suspected-data list:
  - session dropouts
  - wrong arithmetic
  - wrong phone numbers
  - missed required actions
  - clear state contradictions

## Output expectations

- Put outputs under `results/consistency_<timestamp>_<model>_review/` by default when running fresh.
- For existing comparison dirs, default to `results/consistency_<timestamp>_review/`.
- Always produce a polished HTML file.
- The HTML should be framed as a candidate review, not an authoritative final verdict.
- The script should also emit a review packet with enough raw evidence for manual inspection.
- In the final response, summarize:
  - overall score variance per benchmark
  - strongest candidate data/judge issues
  - which ones were manually confirmed
  - where the HTML file lives

## Notes

- If a run folder is partial or missing judged outputs, do not include it in the final comparison set.
- If a benchmark already has completed judged runs but no comparison folder, rebuild the comparison folder rather than rerunning the benchmark.
- When the HTML is the main deliverable, keep it self-contained with embedded CSS and no external assets.
