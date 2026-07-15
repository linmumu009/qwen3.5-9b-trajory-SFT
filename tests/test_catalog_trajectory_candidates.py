from pathlib import Path
import json
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from catalog_trajectory_candidates import canonical_tools  # noqa: E402


def call(name, arguments):
    return {
        "role": "tool_call",
        "content": json.dumps({"name": name, "arguments": arguments}),
    }


def test_canonical_tools_validate_argument_values():
    assert canonical_tools([call("bash", {"command": "ls", "timeout": 10})]) == (True, None)
    assert canonical_tools([call("bash", {"command": "ls", "timeout": 0})]) == (
        False,
        "invalid_argument_value",
    )
    assert canonical_tools([call("read", {"path": "a", "offset": -1})]) == (
        False,
        "invalid_argument_value",
    )


def test_edit_requires_a_complete_replacement():
    assert canonical_tools([call("edit", {"path": "a", "oldText": "x", "newText": "y"})]) == (
        True,
        None,
    )
    assert canonical_tools(
        [call("edit", {"path": "a", "edits": [{"oldText": "x", "newText": "y"}]})]
    ) == (True, None)
    assert canonical_tools([call("edit", {"path": "a"})]) == (
        False,
        "invalid_argument_value",
    )
