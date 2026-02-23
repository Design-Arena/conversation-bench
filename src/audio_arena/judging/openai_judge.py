"""
OpenAI-based transcript judge (realignment + over-clarification handling).

Mirrors the Claude judge but uses OpenAI's chat completions API.
Shares the same system prompt and evaluation methodology.

Usage via CLI:
    uv run audio-arena judge runs/grocery_bench/20251215T202910_gpt-4o-... --judge openai
    uv run audio-arena judge runs/... --judge openai --judge-model o3
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from .llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    format_turns_for_judge,
    load_transcript,
    write_outputs,
)


OPENAI_JUDGE_VERSION = "openai-v1-state-absorbs-tool-penalty"
OPENAI_JUDGE_MODEL = "gpt-5.2"


async def judge_with_openai(
    run_dir: Path,
    only_turns: Optional[set[int]] = None,
    debug: bool = False,
    expected_turns: Optional[List[Dict[str, Any]]] = None,
    skip_turn_taking: bool = False,
    get_relevant_dimensions_fn=None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Main judging function using OpenAI, with two-phase realignment approach.

    Args:
        run_dir: Path to the run directory containing transcript.jsonl
        only_turns: Optional set of turn indices to judge
        debug: Enable debug logging
        expected_turns: Optional list of expected turns. If not provided, imports from turns module.
        skip_turn_taking: If True, skip turn-taking analysis
        get_relevant_dimensions_fn: Function to get relevant scoring dimensions for a turn.
        model: OpenAI model to use. Defaults to OPENAI_JUDGE_MODEL.

    Returns:
        Dict with judgments, realignment_notes, function_tracking, turn_taking_analysis, summary, and model_name.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai package not installed.", file=sys.stderr)
        print("Install with: uv add openai", file=sys.stderr)
        sys.exit(1)

    judge_model = model or OPENAI_JUDGE_MODEL

    records = load_transcript(run_dir)

    if expected_turns is None:
        from benchmarks.conversation_bench.turns import turns as expected_turns

    if only_turns is not None:
        records = [r for r in records if r["turn"] in only_turns]

    if not records:
        raise ValueError("No turns to judge")

    model_name = records[0].get("model_name", "unknown")

    if debug:
        print(f"Judging {len(records)} turns with OpenAI ({judge_model})...", file=sys.stderr)

    # Run turn-taking analysis if WAV file exists
    turn_taking_data: Optional[Dict[int, Dict[str, Any]]] = None
    turn_taking_analysis = None
    if not skip_turn_taking:
        wav_path = run_dir / "conversation.wav"
        if wav_path.exists():
            if debug:
                print("Running turn-taking analysis...", file=sys.stderr)
            try:
                from .turn_taking import analyze_turn_taking
                turn_taking_analysis = analyze_turn_taking(run_dir)
                if turn_taking_analysis.error:
                    if debug:
                        print(f"Turn-taking analysis error: {turn_taking_analysis.error}", file=sys.stderr)
                else:
                    turn_taking_data = {
                        idx: result.to_dict()
                        for idx, result in turn_taking_analysis.per_turn.items()
                    }
                    if debug and turn_taking_analysis.failed_turns:
                        print(f"Turn-taking failures: {turn_taking_analysis.failed_turns}", file=sys.stderr)
            except Exception as e:
                if debug:
                    print(f"Turn-taking analysis failed: {e}", file=sys.stderr)

    formatted_turns = format_turns_for_judge(
        records, expected_turns, only_turns, turn_taking_data, get_relevant_dimensions_fn
    )

    prompt = f"""{formatted_turns}

Please perform your two-phase evaluation:
1. First, analyze each turn against its golden expectation
2. Then, identify any turn misalignments (early/late function calls)
3. Apply realignment adjustments to avoid double-penalizing
4. Output the final JSON with judgments for ALL {len(records)} turns

CRITICAL: Your final_judgments array MUST contain exactly {len(records)} entries (turns 0-{len(records)-1}).

Remember:
- If a function is called early (before expected turn), subsequent turns should not be penalized for the "missing" call
- If a function is called late, credit the turn that did call it (tool_use_correct=TRUE). For the turn that should have called: if they **over-clarified** and ambiguity_handling is in Score Dimensions → tool_use_correct=TRUE, ambiguity_handling=FALSE; if they **forgot state** and state_tracking is in Score Dimensions → tool_use_correct=TRUE, state_tracking=FALSE; if neither dimension can absorb → tool_use_correct=FALSE; if they asked for unnecessary confirmation → tool_use_correct=FALSE
- **Penalty absorption rule**: When a tool call is missed due to a more specific root cause, the penalty lands on the specific dimension (ambiguity_handling or state_tracking) if it's in Score Dimensions. If the specific dimension is NOT in Score Dimensions, fall back to tool_use_correct=FALSE. The penalty must always land somewhere.
- Missing/wrong tool call (not over-clarification or state failure) → tool_use_correct=FALSE only; do not fail instruction_following
- Words contradict actions (e.g. says "I'll wait" but calls in same turn) → tool_use_correct=FALSE and instruction_following=FALSE
- Be generous with kb_grounding unless there's a clear factual error
- Empty assistant_text with a valid tool call is still a valid turn - evaluate the tool call
"""

    client = AsyncOpenAI()

    # o3 / o-series models use developer messages instead of system messages
    is_o_series = judge_model.startswith("o")

    messages = []
    if is_o_series:
        messages.append({"role": "developer", "content": JUDGE_SYSTEM_PROMPT})
    else:
        messages.append({"role": "system", "content": JUDGE_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict[str, Any] = {
        "model": judge_model,
        "messages": messages,
    }

    if is_o_series:
        kwargs["reasoning_effort"] = "high"
        kwargs["response_format"] = {"type": "json_object"}
    else:
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["temperature"] = 0

    if debug:
        print(f"Sending request to OpenAI ({judge_model})...", file=sys.stderr)

    response = await client.chat.completions.create(**kwargs)

    response_text = response.choices[0].message.content or ""

    if debug:
        print(f"OpenAI response length: {len(response_text)} chars", file=sys.stderr)
        if response.usage:
            print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion", file=sys.stderr)

    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1

    if json_start == -1 or json_end == 0:
        raise ValueError(f"No JSON found in response: {response_text[:500]}")

    json_str = response_text[json_start:json_end]

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parse error: {e}", file=sys.stderr)
            print(f"Attempted to parse: {json_str[:500]}...", file=sys.stderr)
        raise ValueError(f"Failed to parse JSON response: {e}")

    final_judgments = result.get('final_judgments', [])
    realignment_notes = result.get('realignment_notes', '')
    function_tracking = result.get('function_call_tracking', {})

    if debug:
        print(f"\nRealignment notes: {realignment_notes}", file=sys.stderr)
        print(f"Function tracking: {json.dumps(function_tracking, indent=2)}", file=sys.stderr)

    judgments = {}
    for j in final_judgments:
        turn_num = j.get('turn')
        if turn_num is not None:
            turn_taking = j.get('turn_taking', True)

            if turn_taking_data and turn_num in turn_taking_data:
                turn_taking = turn_taking_data[turn_num].get('turn_taking', True)

            ambiguity = j.get('ambiguity_handling')
            state = j.get('state_tracking')

            judgments[turn_num] = {
                "scores": {
                    "turn_taking": turn_taking,
                    "tool_use_correct": j.get('tool_use_correct'),
                    "instruction_following": j.get('instruction_following', False),
                    "kb_grounding": j.get('kb_grounding', False),
                    "ambiguity_handling": ambiguity,
                    "state_tracking": state,
                },
                "reasoning": j.get('reasoning', ''),
            }

            if turn_taking_data and turn_num in turn_taking_data:
                issues = turn_taking_data[turn_num].get('issues', [])
                if issues:
                    judgments[turn_num]["turn_taking_issues"] = issues

    expected_turn_numbers = {r["turn"] for r in records}
    judged_turn_numbers = set(judgments.keys())
    missing = expected_turn_numbers - judged_turn_numbers

    if missing:
        raise ValueError(
            f"Failed to get judgments for turns: {sorted(missing)}. "
            f"Expected {len(expected_turn_numbers)} judgments, got {len(judgments)}."
        )

    return {
        "judgments": judgments,
        "realignment_notes": realignment_notes,
        "function_tracking": function_tracking,
        "turn_taking_analysis": turn_taking_analysis.to_dict() if turn_taking_analysis else None,
        "summary": f"Evaluated {len(judgments)} turns with realignment.",
        "model_name": model_name,
        "judge_model": judge_model,
    }
