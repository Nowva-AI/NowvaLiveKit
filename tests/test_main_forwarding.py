"""main.py must relay every pipeline message the voice agent's coaching service handles.

Read from source: importing main.py starts the database and the voice stack.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
# Handled by the coaching service only to log that it is ignored.
DELIBERATELY_UNFORWARDED = {"set_complete"}


def _forwarded_types() -> set[str]:
    tree = ast.parse((SRC / "main.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COACHING_FORWARD_TYPES" for target in node.targets
        ):
            return {element.value for element in node.value.args[0].elts}
    raise AssertionError("COACHING_FORWARD_TYPES not found in main.py")


def _pipeline_sent_types() -> set[str]:
    sent: set[str] = set()
    for path in (SRC / "biomechanics").rglob("*.py"):
        sent |= set(re.findall(r'"type":\s*"([a-z_]+)"', path.read_text()))
    return sent


def _coaching_handled_types() -> set[str]:
    source = (SRC / "agent" / "services" / "coaching_service.py").read_text()
    return set(re.findall(r'msg_type == "([a-z_]+)"', source))


class TestCoachingForwarding:
    def test_every_handled_pipeline_message_is_forwarded(self):
        missing = (_pipeline_sent_types() & _coaching_handled_types()) - _forwarded_types() - DELIBERATELY_UNFORWARDED
        assert missing == set()

    def test_depth_rejected_reps_reach_the_agent(self):
        """A rep the depth gate refuses is announced only as shallow_rep."""
        assert "shallow_rep" in _forwarded_types()
