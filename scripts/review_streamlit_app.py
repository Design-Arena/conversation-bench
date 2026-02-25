import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_experiment_review import SCORE_COLUMNS, build_review_payload, turn_audio_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a review directory, review_payload.json, or judged run directory.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def find_default_input_path() -> str:
    results_root = Path.cwd() / "results"
    payload_paths = sorted(results_root.glob("**/review_payload.json"))
    if payload_paths:
        return str(payload_paths[-1].parent)

    runs_root = Path.cwd() / "runs"
    judged_paths = sorted(runs_root.glob("**/openai_judged.jsonl"))
    if judged_paths:
        return str(judged_paths[-1].parent)

    return ""


def load_review_source(input_path: str) -> tuple[dict, str, str]:
    candidate_path = Path(input_path).expanduser().resolve()
    if candidate_path.is_file():
        if candidate_path.name != "review_payload.json":
            raise FileNotFoundError(
                "Expected a review_payload.json file when using a file path."
            )
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        review_dir = candidate_path.parent
        analysis_path = review_dir / "error_analysis.md"
        error_analysis = (
            analysis_path.read_text(encoding="utf-8")
            if analysis_path.exists()
            else "No saved error analysis found next to review_payload.json."
        )
        return payload, error_analysis, str(candidate_path)

    if candidate_path.is_dir():
        payload_path = candidate_path / "review_payload.json"
        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            analysis_path = candidate_path / "error_analysis.md"
            error_analysis = (
                analysis_path.read_text(encoding="utf-8")
                if analysis_path.exists()
                else "No saved error analysis found next to review_payload.json."
            )
            return payload, error_analysis, str(payload_path)

        judged_path = candidate_path / "openai_judged.jsonl"
        if judged_path.exists():
            payload, _, error_analysis, _ = build_review_payload(
                candidate_path,
                output_dir=candidate_path,
            )
            return payload, error_analysis, str(candidate_path)

    raise FileNotFoundError(
        "Input must be a review directory, a review_payload.json file, or a judged run directory."
    )


def apply_filters(
    rows_df: pd.DataFrame,
    *,
    dimension: str,
    status: str,
    response_status: str,
    search_text: str,
) -> pd.DataFrame:
    filtered_df = rows_df.copy()

    if response_status != "all":
        filtered_df = filtered_df[filtered_df["response_status"] == response_status]

    if dimension == "all":
        if status != "all":
            filtered_df = filtered_df[filtered_df["overall_status"] == status]
    else:
        dimension_values = filtered_df[dimension]
        dimension_status = pd.Series("incomplete", index=filtered_df.index)
        dimension_status = dimension_status.mask(dimension_values == True, "pass")
        dimension_status = dimension_status.mask(dimension_values == False, "fail")
        if status != "all":
            filtered_df = filtered_df[dimension_status == status]

    needle = search_text.strip().lower()
    if needle:
        search_columns = [
            "user_text",
            "assistant_text",
            "judge_reasoning",
            "failed_dimensions_text",
            "tool_calls_text",
            "tool_results_text",
        ]
        for optional_column in [
            "golden_tool_calls_text",
            "golden_tool_results_text",
            "golden_context_block",
        ]:
            if optional_column in filtered_df.columns:
                search_columns.append(optional_column)
        haystack = filtered_df[search_columns].fillna("").astype(str).agg(
            " ".join,
            axis=1,
        )
        haystack = haystack + " " + filtered_df["turn"].astype(str)
        filtered_df = filtered_df[haystack.str.lower().str.contains(needle, regex=False)]

    status_order = {"fail": 0, "incomplete": 1, "pass": 2}
    filtered_df = filtered_df.assign(
        _status_order=filtered_df["overall_status"].map(status_order).fillna(3)
    )
    filtered_df = filtered_df.sort_values(
        by=["_status_order", "failure_count", "turn"],
        ascending=[True, False, True],
    ).drop(columns="_status_order")
    return filtered_df.reset_index(drop=True)


def format_previous_turns(previous_rows: pd.DataFrame) -> str:
    if previous_rows.empty:
        return "No previous turns."

    parts: list[str] = []
    for _, row in previous_rows.iterrows():
        context_block = row.get("golden_context_block")
        if context_block:
            parts.append(str(context_block))
            continue
        parts.append(
            "\n".join(
                [
                    f"Turn {int(row['turn'])}",
                    f"User: {row.get('golden_user_text', row['user_text'])}",
                    f"Assistant: {row.get('golden_assistant_text', row['assistant_text'])}",
                ]
            )
        )
    return "\n\n".join(parts)


def load_audio_bytes(audio_path: Path) -> bytes | None:
    if not audio_path.exists():
        return None
    return audio_path.read_bytes()


def main() -> None:
    cli_args = parse_args()
    default_input_path = cli_args.input_path or find_default_input_path()

    st.set_page_config(
        page_title="ConversationBench Review",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ConversationBench Review App")
    st.caption(
        "Load a generated review directory or a judged run directory. The app keeps the selected turn in session state and shows all prior turns for context."
    )

    input_path = st.sidebar.text_input(
        "Review directory, review payload, or judged run directory",
        value=default_input_path,
    )

    if not input_path.strip():
        st.info(
            "Provide a path in the sidebar. You can point to a generated review directory, review_payload.json, or a judged run directory."
        )
        return

    try:
        payload, error_analysis, source_path = load_review_source(input_path)
    except Exception as exc:
        st.error(str(exc))
        return

    st.sidebar.caption(f"Loaded from: {source_path}")

    rows_df = pd.DataFrame(payload["rows"]).sort_values("turn").reset_index(drop=True)
    if rows_df.empty:
        st.warning("No rows found in the loaded payload.")
        return

    dimension = st.sidebar.selectbox(
        "Dimension",
        options=["all", *SCORE_COLUMNS],
        index=0,
    )
    status = st.sidebar.selectbox(
        "Status",
        options=["all", "fail", "pass", "incomplete"],
        index=0,
    )
    response_status = st.sidebar.selectbox(
        "Response status",
        options=["all", "normal", "empty_response"],
        index=0,
    )
    search_text = st.sidebar.text_input("Search")

    filtered_df = apply_filters(
        rows_df,
        dimension=dimension,
        status=status,
        response_status=response_status,
        search_text=search_text,
    )

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    turn_options = filtered_df["turn"].astype(int).tolist()
    if (
        "selected_turn" not in st.session_state
        or st.session_state["selected_turn"] not in turn_options
    ):
        st.session_state["selected_turn"] = turn_options[0]

    st.sidebar.selectbox(
        "Selected turn",
        options=turn_options,
        key="selected_turn",
    )
    selected_turn = int(st.session_state["selected_turn"])

    summary_columns = st.columns(6)
    summary_columns[0].metric(
        "Rows",
        str(payload["metadata"]["turn_count"]),
        f"Filtered {len(filtered_df)}",
    )
    summary_columns[1].metric(
        "Overall Pass Rows",
        str(payload["summary"]["overall_pass_rows"]),
    )
    summary_columns[2].metric(
        "Empty Responses",
        str(payload["summary"]["empty_response_count"]),
    )
    summary_columns[3].metric(
        "Instruction",
        str(payload["summary"]["passes"]["instruction_following"]),
    )
    summary_columns[4].metric(
        "Grounding",
        str(payload["summary"]["passes"]["kb_grounding"]),
    )
    summary_columns[5].metric(
        "Tool Use",
        str(payload["summary"]["passes"]["tool_use_correct"]),
    )

    st.subheader("Run Metadata")
    metadata_df = pd.DataFrame(
        {
            "field": list(payload["metadata"].keys()),
            "value": [str(value) for value in payload["metadata"].values()],
        }
    )
    st.dataframe(
        metadata_df,
        use_container_width=True,
        hide_index=True,
    )

    selected_row = rows_df[rows_df["turn"] == selected_turn].iloc[0]
    previous_rows = rows_df[rows_df["turn"] < selected_turn].sort_values("turn")
    previous_turns_text = format_previous_turns(previous_rows)
    benchmark_name = payload["metadata"]["benchmark_name"]
    selected_turn_audio_path = turn_audio_path(benchmark_name, selected_turn)
    selected_turn_audio_bytes = load_audio_bytes(selected_turn_audio_path)
    conversation_audio_path = Path(
        payload["metadata"].get(
            "conversation_audio_path",
            str(Path(payload["metadata"]["source_run_dir"]) / "conversation.wav"),
        )
    )
    conversation_audio_bytes = load_audio_bytes(conversation_audio_path)

    detail_tab, curated_tab, full_tab, summary_tab, analysis_tab = st.tabs(
        [
            "Selected Turn",
            "Curated Failures",
            "All Datapoints",
            "Dimension Summary",
            "Error Analysis",
        ]
    )

    with detail_tab:
        st.subheader(f"Turn {selected_turn}")
        with st.expander(
            f"Gold previous turns context ({len(previous_rows)} prior turns)",
            expanded=True,
        ):
            st.text_area(
                "Conversation history",
                value=previous_turns_text,
                height=420,
                disabled=True,
                label_visibility="collapsed",
            )

        st.markdown("**Current turn audio**")
        if selected_turn_audio_bytes is None:
            st.info("No per-turn input WAV found for this turn.")
        else:
            st.audio(selected_turn_audio_bytes, format="audio/wav")

        st.markdown("**Full conversation audio**")
        if conversation_audio_bytes is None:
            st.info("No run-level conversation.wav found.")
        else:
            st.audio(conversation_audio_bytes, format="audio/wav")

        content_col, detail_col = st.columns([3, 2])

        with content_col:
            st.markdown("**User query**")
            st.write(selected_row["user_text"])
            st.markdown("**Model output**")
            st.write(selected_row["assistant_text"])
            st.markdown("**Judge reasoning**")
            st.write(selected_row["judge_reasoning"])

        with detail_col:
            compact_metadata_df = pd.DataFrame(
                [
                    ["timestamp", str(selected_row["timestamp"])],
                    ["overall_status", str(selected_row["overall_status"])],
                    ["response_status", str(selected_row["response_status"])],
                    ["latency_ms", str(selected_row["latency_ms"])],
                    ["ttfb_ms", str(selected_row["ttfb_ms"])],
                    ["reconnection_count", str(selected_row["reconnection_count"])],
                    ["failed_dimensions", str(selected_row["failed_dimensions_text"] or "none")],
                ],
                columns=["field", "value"],
            )
            st.markdown("**Compact metadata**")
            st.dataframe(
                compact_metadata_df,
                use_container_width=True,
                hide_index=True,
            )

            score_rows: list[dict] = []
            for column in SCORE_COLUMNS:
                value = selected_row[column]
                if pd.isna(value):
                    verdict = "n/a"
                elif bool(value):
                    verdict = "pass"
                else:
                    verdict = "fail"
                score_rows.append({"dimension": column, "verdict": verdict})
            st.markdown("**Per-dimension verdicts**")
            st.dataframe(
                pd.DataFrame(score_rows),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Tool calls", expanded=False):
                st.json(json.loads(selected_row["tool_calls_text"]))
            with st.expander("Tool results", expanded=False):
                st.json(json.loads(selected_row["tool_results_text"]))

    with curated_tab:
        curated_df = filtered_df[filtered_df["overall_status"] == "fail"].copy()
        curated_df = curated_df.sort_values("turn").head(25)
        st.caption(
            f"Showing {len(curated_df)} of {int((filtered_df['overall_status'] == 'fail').sum())} failing rows after filters. Sorted by turn."
        )
        if curated_df.empty:
            st.info("No failing rows match the current filters.")
        else:
            st.dataframe(
                curated_df[
                    [
                        "turn",
                        "overall_status",
                        "failed_dimensions_text",
                        "user_text",
                        "assistant_text",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=520,
            )

    with full_tab:
        st.caption(
            f"Showing {len(filtered_df)} of {len(rows_df)} rows. Full table keeps failures first, then sorts by severity and turn."
        )
        st.dataframe(
            filtered_df[
                [
                    "turn",
                    "overall_status",
                    "response_status",
                    "latency_ms",
                    "failed_dimensions_text",
                    "user_text",
                    "assistant_text",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=620,
        )

    with summary_tab:
        st.caption("Pass rate is computed over applicable rows for each dimension.")
        st.dataframe(
            pd.DataFrame(payload["dimension_summary"]),
            use_container_width=True,
            hide_index=True,
        )

    with analysis_tab:
        st.markdown(error_analysis)


if __name__ == "__main__":
    main()
