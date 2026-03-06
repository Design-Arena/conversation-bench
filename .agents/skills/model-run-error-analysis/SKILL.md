---
name: model-run-error-analysis
description: Analyze judged benchmark runs to identify dominant error modes per model, shared hard turns, and representative failing examples. Use when asked for a deep error analysis across one or more model runs, especially from a multi-model comparison folder or a set of `openai_judged.jsonl` run directories.
---

# Model Run Error Analysis

## Overview

Use this skill to produce a model-by-model failure analysis from judged benchmark runs.

The key requirement is that the final buckets must come from **deep datapoint review**, not just aggregate metrics or heuristic labels.

The skill is designed for:
- multi-model comparison folders produced by [compare_model_runs.py](/Users/minh.hoque/work/github/conversation-bench/scripts/compare_model_runs.py)
- one or more judged run directories containing `openai_judged.jsonl`

The output is a markdown report under `results/` that explains:
- dominant failed dimensions per model
- major error-mode clusters
- a plain-language description of each error mode per model
- representative failing turns
- shared failures across models
- model-specific unique failures and unique passes

The skill also exports a failure review packet with raw row content so the model can inspect:
- input
- model output
- grader verdicts
- grader reasoning
- tool calls
- tool results
- per-dimension scores

## Preferred entrypoint

Run:

```bash
python3 /Users/minh.hoque/work/github/conversation-bench/.agents/skills/model-run-error-analysis/scripts/deep_error_analysis.py \
  --comparison-dir /absolute/path/to/comparison_dir
```

Or:

```bash
python3 /Users/minh.hoque/work/github/conversation-bench/.agents/skills/model-run-error-analysis/scripts/deep_error_analysis.py \
  --run-dir /absolute/path/to/run_a \
  --run-dir /absolute/path/to/run_b \
  --run-dir /absolute/path/to/run_c
```

## Inputs

Provide one of:
- `--comparison-dir <dir>` where the directory contains `run_results.csv`
- one or more `--run-dir <dir>` paths where each directory contains `openai_judged.jsonl`

Optional:
- `--output <path>` to override the report destination

## Workflow

This skill has two stages.

### Stage 1: Build the review packet

1. Load each run’s `openai_judged.jsonl`.
2. For every turn, extract:
   - `turn`
   - `user_text`
   - `assistant_text`
   - `tool_calls`
   - `tool_results`
   - `scores`
   - `judge_reasoning`
3. Mark a row as failed if any score dimension is `false`.
4. Export failure review rows to JSONL and CSV so the model can inspect raw datapoints directly.
5. Compute exact per-model failure counts:
   - failed rows
   - `MODEL_ENDED_SESSION`
   - `EMPTY_RESPONSE`
   - failed dimensions
6. Generate provisional heuristic clusters from the judge reasoning and response shape:
   - premature end session
   - empty response
   - partial multi-tool execution
   - state memory failure
   - ambiguity handling failure
   - knowledge grounding error
   - tool or action selection error
   - other
7. Compare turn-level failure sets across models:
   - failed by all models
   - unique fail turns per model
   - unique pass turns per model

### Stage 2: Deep review and final bucketing

8. Review actual failed datapoints, not just the heuristic labels.
9. For each provisional bucket, inspect representative rows and confirm the real mechanism by reading:
   - user input
   - assistant output
   - full `judge_reasoning`
   - `tool_calls`
   - `tool_results`
   - all failed dimensions
10. Re-bucket when the heuristic label is too coarse or wrong.
11. Distinguish:
   - the true underlying model mistake
   - secondary grader-visible consequences

Example:
- if the model ends the session after one correct tool call in a 3-step workflow, the real error is usually `premature_end_session` or `partial_multi_tool_execution`, not just `instruction_following=false`
- if the model asks for the user name again and misses a tool call, the real bucket is often `state memory failure`, even if `tool_use_correct` is absorbed

12. Use the provisional clusters only to accelerate review. They are not the final truth.
13. Write `deep_error_analysis.md` with buckets that reflect the underlying error mechanism.
14. In the final assistant response, also provide a concise plain-text summary of the findings:
   - main error modes per model
   - the most important shared hard turns or shared failure themes
   - the path to the saved markdown report

## Output expectations

- Put the report under `results/` by default.
- Also write:
  - `failure_review_rows.jsonl`
  - `failure_review_rows.csv`
- Also return a text summary to the user in the same turn. Do not make the markdown file the only deliverable.
- Use exact dimension counts from judged rows.
- Treat broader “error modes” as analyst/script-defined clusters, not as a second judge.
- Cite representative turn numbers in the report.
- For each model, describe each major error mode in plain language:
  - what the model is doing wrong
  - what kinds of turns trigger it
  - what grader-visible dimensions it usually causes
- For each major error mode, include concrete datapoints to review quickly:
  - turn number
  - user input
  - assistant output
  - judge reasoning
  - tool calls / tool results when relevant
- Prefer comparing models on:
  - dominant failure dimensions
  - response-pattern failures
  - shared hard turns
  - unique deltas
- In the final report, prefer buckets that explain the **cause** of failure, not just the surface grader dimension.

## Guardrails

- Do not infer error modes from aggregate pass-rate charts alone.
- Use `openai_judged.jsonl` as the source of truth.
- Read actual `tool_calls` and `tool_results` for tool-failure turns.
- When a model is fluent but wrong, classify by the decisive failure mechanism, not by tone.
- Do not let the heuristic classifier become the final report unreviewed.
- For each top bucket in the final writeup, manually inspect multiple concrete turns before naming the bucket.
- If both a comparison dir and explicit run dirs are given, prefer the explicit run dirs.
