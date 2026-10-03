"""
main.py tells the display which biomechanics profile runs, so it can pick that
profile's score dimensions. Read from source: importing main.py starts the
database and the voice stack.
"""

from __future__ import annotations

import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2] / "src" / "main.py"


def _workout_start_events() -> list[ast.Dict]:
    events = []
    for node in ast.walk(ast.parse(MAIN.read_text())):
        if not isinstance(node, ast.Dict):
            continue
        literal = {
            key.value: value.value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        }
        if literal.get("type") == "workout" and literal.get("action") == "start":
            events.append(node)
    return events


class TestDisplayProfile:
    def test_every_workout_start_event_names_the_profile(self):
        events = _workout_start_events()
        assert len(events) == 2
        for event in events:
            assert "profile" in {key.value for key in event.keys if isinstance(key, ast.Constant)}

    def test_the_profile_is_resolved_like_the_pose_process_does(self):
        source = MAIN.read_text()
        assert "from biomechanics.profiles import get_profile" in source
