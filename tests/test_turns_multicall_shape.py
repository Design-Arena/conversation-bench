import ast
from pathlib import Path


def _load_turns_from_file(path: Path) -> list[dict]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "turns":
                    return ast.literal_eval(node.value)
    raise AssertionError(f"No turns assignment found in {path}")


def test_multicall_turns_use_list_responses():
    turn_files = [
        Path("benchmarks/appointment_bench/turns.py"),
        Path("benchmarks/event_bench/turns.py"),
        Path("benchmarks/grocery_bench/turns.py"),
    ]

    for turn_file in turn_files:
        turns = _load_turns_from_file(turn_file)
        for idx, turn in enumerate(turns):
            required = turn.get("required_function_call")
            response = turn.get("function_call_response")
            if isinstance(required, list):
                assert isinstance(
                    response, list
                ), f"{turn_file}: turn {idx} has list required_function_call but non-list function_call_response"
                assert len(response) == len(
                    required
                ), f"{turn_file}: turn {idx} has {len(required)} required calls but {len(response)} responses"
