"""
Pure formatters that turn persisted biomechanics data into short text
the coaching LLM can relay verbatim: greeting context, recap comparisons,
and the main-menu progress report. No DB access here.
"""

from __future__ import annotations

MIN_DIMENSION_DELTA_POINTS = 3
TREND_MIN_SETS = 4
RECENT_WINDOW_SETS = 3

DIMENSION_LABELS = {
    "depth": "depth",
    "trunk_control": "trunk control",
    "knee_tracking": "knee tracking",
    "symmetry": "symmetry",
    "tempo": "tempo",
}

FAULT_LABELS = {
    "knee_valgus": "knees caving in",
    "hip_shoot": "hips rising before the chest",
    "heel_rise": "heels lifting",
    "balance": "weight drifting off the middle of the foot",
    "hip_shift": "hips sliding to one side",
    "bilateral_asymmetry": "bar tilting to one side",
    "depth": "not reaching their depth target",
    "foot_placement": "feet set up unevenly",
    "lockout": "not standing all the way up between reps",
    "tempo": "dropping too fast on the way down",
    "depth_drift": "reps getting shallower",
    "velocity_loss": "reps slowing down",
    "forward_lean": "leaning forward",
    # Deadlift
    "deadlift_bar_position": "bar not over the middle of the foot at the start",
    "deadlift_setup_hips": "hips set too high or too low at the start",
    "deadlift_shoulders_behind": "shoulders behind the bar at the start",
    "deadlift_hips_shoot": "hips rising before the chest off the floor",
    "deadlift_bar_drift": "bar drifting away from the legs",
    "deadlift_lockout": "not finishing standing tall",
    "deadlift_lean_back": "leaning back at the top",
    "deadlift_hip_shift": "hips sliding to one side",
    "deadlift_bar_tilt": "bar tilting to one side",
    "deadlift_bent_arms": "arms bending during the pull",
    "deadlift_velocity_loss": "bar slowing down",
}

# Progress is stored per exercise profile ("squat", "deadlift").
DEFAULT_PROGRESS_EXERCISE = "squat"


def _days_ago_phrase(days_ago: int) -> str:
    if days_ago == 0:
        return "earlier today"
    if days_ago == 1:
        return "yesterday"
    return f"{days_ago} days ago"


def fault_label(fault_type: str) -> str:
    return FAULT_LABELS.get(fault_type, fault_type.replace("_", " "))


def _exercise_words(exercise: str) -> str:
    return exercise.replace("_", " ")


def _weakest_dimension(per_dimension: dict) -> tuple[str, float] | None:
    if not per_dimension:
        return None
    key = min(per_dimension, key=per_dimension.get)
    return DIMENSION_LABELS.get(key, key), per_dimension[key]


def build_greeting_progress_line(
    baseline: dict | None, exercise: str = DEFAULT_PROGRESS_EXERCISE,
) -> str | None:
    """One compact context block for the workout greeting."""
    if not baseline:
        return None
    parts = [
        f"PROGRESS CONTEXT: last {_exercise_words(exercise)} session was "
        f"{_days_ago_phrase(baseline.get('days_ago', 0))}"
    ]
    if baseline.get("mean_score") is not None:
        parts.append(f"form score {round(baseline['mean_score'] * 100)} out of 100")
    if baseline.get("total_reps"):
        parts.append(
            f"{baseline['total_reps']} reps across {baseline.get('total_sets', 0)} sets"
        )
    weakest = _weakest_dimension(baseline.get("per_dimension") or {})
    if weakest is not None:
        label, value = weakest
        parts.append(f"weakest area was {label} ({round(value * 100)} out of 100)")
    if baseline.get("avg_knee_valgus_deg") is not None:
        parts.append(
            f"knees caved about {baseline['avg_knee_valgus_deg']} degrees on average"
        )
    top_faults = baseline.get("top_faults") or []
    if top_faults:
        parts.append(f"most common fault: {fault_label(top_faults[0]['fault_type'])}")
    return (
        ", ".join(parts) + ". "
        "Weave ONE brief reference to this into the greeting as today's goal, "
        "in your own words — one quick beat naming what you'll clean up today, "
        "phrased differently every session."
    )


def build_progress_comparison_lines(
    baseline: dict | None, scoring: dict | None
) -> list[str]:
    """Precomputed vs-last-session deltas for the set recap prompt."""
    if not baseline or not scoring:
        return []
    lines: list[str] = []

    # Deltas derive from the rounded endpoints so the spoken numbers
    # always agree ("72 -> 78" is always "+6")
    base_mean = baseline.get("mean_score")
    current_mean = scoring.get("mean_score")
    if base_mean is not None and current_mean is not None:
        base_points = round(base_mean * 100)
        current_points = round(current_mean * 100)
        lines.append(
            f"\nVS LAST SESSION ({_days_ago_phrase(baseline.get('days_ago', 0))}): "
            f"overall {base_points} to {current_points} "
            f"({current_points - base_points:+d} points)"
        )

    base_dims = baseline.get("per_dimension") or {}
    current_dims = scoring.get("per_dimension") or {}
    dim_lines = []
    for key, label in DIMENSION_LABELS.items():
        base_value = base_dims.get(key)
        current_value = current_dims.get(key)
        if base_value is None or current_value is None:
            continue
        base_points = round(base_value * 100)
        current_points = round(current_value * 100)
        if abs(current_points - base_points) >= MIN_DIMENSION_DELTA_POINTS:
            dim_lines.append(
                f"{label}: {base_points} to {current_points} "
                f"({current_points - base_points:+d})"
            )
    if dim_lines:
        lines.append("Changed vs last session: " + "; ".join(dim_lines) + ".")

    if lines:
        lines.append(
            "If something improved vs last session, call it out specifically — "
            "this is the payoff of their training."
        )
    return lines


def build_session_comparison_line(
    baseline: dict | None, session_mean_score: float | None
) -> str | None:
    """One-line whole-session comparison for the exercise recap."""
    if not baseline or session_mean_score is None:
        return None
    base_mean = baseline.get("mean_score")
    if base_mean is None:
        return None
    base_points = round(base_mean * 100)
    current_points = round(session_mean_score * 100)
    delta_points = current_points - base_points
    when = _days_ago_phrase(baseline.get("days_ago", 0))
    if delta_points > 0:
        return (
            f"Session-over-session: {delta_points} points better than last session "
            f"({when}: {base_points}, today: {current_points})."
        )
    if delta_points < 0:
        return (
            f"Session-over-session: {abs(delta_points)} points below last session "
            f"({when}: {base_points}, today: {current_points})."
        )
    return f"Session-over-session: matched last session's form score ({base_points} out of 100)."


def build_progress_report(
    score_rows: list[dict], baseline: dict | None,
    exercise: str = DEFAULT_PROGRESS_EXERCISE,
) -> str:
    """Text progress report for the main-menu view_progress tool."""
    exercise_words = _exercise_words(exercise)
    if not score_rows:
        return (
            f"No {exercise_words} sessions recorded yet. Their progress tracking starts "
            "with their first workout."
        )
    lines = [f"{len(score_rows)} {exercise_words} sets recorded."]

    if baseline:
        summary = f"Last session: {_days_ago_phrase(baseline.get('days_ago', 0))}"
        if baseline.get("mean_score") is not None:
            summary += f", form score {round(baseline['mean_score'] * 100)}/100"
        if baseline.get("total_reps"):
            summary += (
                f", {baseline['total_reps']} reps across "
                f"{baseline.get('total_sets', 0)} sets"
            )
        lines.append(summary + ".")

    def _window_mean(rows: list[dict], key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    early_mean = _window_mean(score_rows[:RECENT_WINDOW_SETS], "mean_score")
    recent_mean = _window_mean(score_rows[-RECENT_WINDOW_SETS:], "mean_score")
    if (
        len(score_rows) >= TREND_MIN_SETS
        and early_mean is not None
        and recent_mean is not None
    ):
        early_points = round(early_mean * 100)
        recent_points = round(recent_mean * 100)
        lines.append(
            f"Form score trend: {early_points} -> {recent_points} "
            f"({recent_points - early_points:+d} points from their first sets to their recent ones)."
        )

    recent_rows = score_rows[-RECENT_WINDOW_SETS:]
    dimension_keys = (
        ("depth", "depth"),
        ("trunk control", "trunk_control"),
        ("knee tracking", "knee_tracking"),
        ("symmetry", "symmetry"),
    )
    current_dims = {}
    for label, key in dimension_keys:
        value = _window_mean(recent_rows, key)
        if value is not None:
            current_dims[label] = value
    if current_dims:
        profile = ", ".join(
            f"{label} {round(value * 100)}" for label, value in current_dims.items()
        )
        lines.append(f"Current form profile (out of 100): {profile}.")
        weakest_label = min(current_dims, key=current_dims.get)
        lines.append(f"Biggest opportunity: {weakest_label}.")

    if baseline and baseline.get("top_faults"):
        top = baseline["top_faults"][0]
        lines.append(
            f"Most common fault last session: {fault_label(top['fault_type'])} "
            f"({top['count']} reps)."
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Multi-session trend formatters
# ------------------------------------------------------------------

MAX_GREETING_SETS = 5
MAX_GREETING_FAULTS = 3


def build_detailed_greeting_context(
    baseline: dict | None, fault_trends: dict | None
) -> str | None:
    """Rich greeting context with per-set last-session detail and chronic fault trends."""
    if not baseline and not fault_trends:
        return None

    parts: list[str] = []

    if baseline:
        header = (
            f"LAST SESSION ({_days_ago_phrase(baseline.get('days_ago', 0))}): "
            f"form score {round(baseline['mean_score'] * 100)}/100, "
            f"{baseline.get('total_reps', 0)} reps across "
            f"{baseline.get('total_sets', 0)} sets"
            if baseline.get("mean_score") is not None
            else f"LAST SESSION ({_days_ago_phrase(baseline.get('days_ago', 0))}): "
            f"{baseline.get('total_reps', 0)} reps across "
            f"{baseline.get('total_sets', 0)} sets"
        )
        parts.append(header)

        per_set = baseline.get("per_set") or []
        if per_set:
            set_lines: list[str] = []
            for s in per_set[:MAX_GREETING_SETS]:
                faults = s.get("faults", {})
                if faults:
                    fault_parts = [
                        f"{fault_label(ft)} on {cnt}" for ft, cnt in faults.items()
                    ]
                    set_lines.append(
                        f"Set {s['set_number']}: {s.get('rep_count', 0)} reps, "
                        + ", ".join(fault_parts)
                    )
                else:
                    set_lines.append(
                        f"Set {s['set_number']}: {s.get('rep_count', 0)} reps, all clean"
                    )
            parts.append("PER-SET DETAIL: " + ". ".join(set_lines) + ".")

    if fault_trends and fault_trends.get("sessions_analyzed", 0) >= 2:
        profile = fault_trends.get("fault_profile", [])[:MAX_GREETING_FAULTS]
        n_sessions = fault_trends["sessions_analyzed"]
        if profile:
            trend_lines: list[str] = []
            for entry in profile:
                label = fault_label(entry["fault_type"])
                trend_suffix = ""
                if entry.get("trend") == "improving":
                    trend_suffix = " (improving)"
                elif entry.get("trend") == "worsening":
                    trend_suffix = " (getting worse)"
                trend_lines.append(
                    f"{label} appeared in {entry['sessions_present']} of "
                    f"{n_sessions} sessions ({entry['total_occurrences']} total reps)"
                    f"{trend_suffix}"
                )
            parts.append("MULTI-SESSION TRENDS: " + ". ".join(trend_lines) + ".")

        chronic = fault_trends.get("chronic_faults", [])
        if chronic:
            labels = [fault_label(ft) for ft in chronic[:MAX_GREETING_FAULTS]]
            parts.append(f"CHRONIC FAULTS TO WATCH: {', '.join(labels)}.")

    if not parts:
        return None

    parts.append(
        "Weave a brief reference to the chronic or most common fault into the "
        "greeting as today's goal. If a fault is improving, acknowledge the progress."
    )
    return " ".join(parts)


def build_chronic_fault_celebration(
    fault_trends: dict | None,
    current_session_faults: dict[str, int],
    completed_sets: int,
) -> str | None:
    """Celebration directive when a chronic fault is absent in the current session."""
    if not fault_trends or completed_sets < 2:
        return None

    chronic = fault_trends.get("chronic_faults", [])
    if not chronic:
        return None

    n_sessions = fault_trends.get("sessions_analyzed", 0)
    profile_lookup = {
        e["fault_type"]: e for e in fault_trends.get("fault_profile", [])
    }

    celebrations: list[str] = []
    for ft in chronic:
        if current_session_faults.get(ft, 0) > 0:
            continue
        entry = profile_lookup.get(ft)
        if not entry:
            continue
        label = fault_label(ft)
        celebrations.append(
            f"CELEBRATE THIS: {label} has been a consistent issue — it appeared "
            f"in {entry['sessions_present']} of your last {n_sessions} sessions — "
            f"but it hasn't shown up at ALL today after {completed_sets} sets. "
            f"This is a real breakthrough — make sure they feel that."
        )

    return " ".join(celebrations) if celebrations else None


def build_trend_comparison_lines(
    fault_trends: dict | None, current_faults: dict[str, int], current_reps: int
) -> list[str]:
    """Cross-session trend context for recaps, compared as per-rep rates."""
    if not fault_trends or current_reps <= 0:
        return []

    n_sessions = fault_trends.get("sessions_analyzed", 0)
    total_reps = fault_trends.get("total_reps", 0)
    if n_sessions < 2 or total_reps <= 0:
        return []

    profile_lookup = {
        e["fault_type"]: e for e in fault_trends.get("fault_profile", [])
    }

    lines: list[str] = []
    for ft, count in current_faults.items():
        entry = profile_lookup.get(ft)
        if not entry or entry["total_occurrences"] < 3:
            continue
        usual_rate = entry["total_occurrences"] / total_reps
        current_rate = count / current_reps
        label = fault_label(ft)
        if current_rate < usual_rate * 0.5:
            lines.append(
                f"CROSS-SESSION TREND: {label} showed on {count} of {current_reps} reps — "
                f"well below their usual rate from recent sessions, trending the right direction."
            )
        elif current_rate > usual_rate * 1.5:
            lines.append(
                f"CROSS-SESSION TREND: {label} showed on {count} of {current_reps} reps — "
                f"more often than their usual rate from recent sessions."
            )

    return lines
