"""YAML graph loader — loads symptoms.yaml and causes.yaml into frozen dicts.

Resolves string function references to actual callables from evidence_tests
and parameter_deltas modules. Validates all cross-references at import time.

Exports:
    SYMPTOM_GRAPH — MappingProxyType of symptom definitions
    CAUSE_GRAPH   — MappingProxyType of cause definitions
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import yaml

from . import evidence_tests, parameter_deltas

_GRAPH_DIR = Path(__file__).parent


def _resolve_function(module, function_name: str) -> Callable:
    func = getattr(module, function_name, None)
    if func is None:
        raise ValueError(
            f"Function '{function_name}' not found in {module.__name__}"
        )
    if not callable(func):
        raise ValueError(
            f"'{function_name}' in {module.__name__} is not callable"
        )
    return func


def _load_yaml(filename: str) -> dict:
    filepath = _GRAPH_DIR / filename
    with open(filepath) as file_handle:
        return yaml.safe_load(file_handle)


def _build_symptom_graph() -> MappingProxyType:
    raw = _load_yaml("symptoms.yaml")
    graph: dict[str, dict[str, Any]] = {}

    for symptom_id, definition in raw.items():
        expected_fn_name = definition["expected_value_fn"]
        expected_fn = _resolve_function(evidence_tests, expected_fn_name)

        candidate_causes = []
        for entry in definition["candidate_causes"]:
            candidate_causes.append(
                MappingProxyType(
                    {"cause_id": entry["cause_id"], "prior": entry["prior"]}
                )
            )

        graph[symptom_id] = MappingProxyType(
            {
                "description": definition["description"],
                "detection": MappingProxyType(definition["detection"]),
                "expected_value_fn": expected_fn,
                "severity_scoring": definition["severity_scoring"],
                "threshold": definition["threshold"],
                "measurement": definition["measurement"],
                "candidate_causes": tuple(candidate_causes),
            }
        )

    return MappingProxyType(graph)


def _build_cause_graph() -> MappingProxyType:
    raw = _load_yaml("causes.yaml")
    graph: dict[str, dict[str, Any]] = {}

    for cause_id, definition in raw.items():
        evidence_fn = _resolve_function(
            evidence_tests, definition["evidence_test_fn"]
        )

        delta_fn_name = definition.get("parameter_delta_fn")
        delta_fn = None
        if delta_fn_name:
            delta_fn = _resolve_function(parameter_deltas, delta_fn_name)

        graph[cause_id] = MappingProxyType(
            {
                "description": definition["description"],
                "tier": definition["tier"],
                "evidence_test_fn": evidence_fn,
                "parameter_delta_fn": delta_fn,
                "explanation_template": definition["explanation_template"],
            }
        )

    return MappingProxyType(graph)


def _validate_cross_references(
    symptom_graph: MappingProxyType, cause_graph: MappingProxyType
) -> None:
    for symptom_id, symptom_def in symptom_graph.items():
        for candidate in symptom_def["candidate_causes"]:
            cause_id = candidate["cause_id"]
            if cause_id not in cause_graph:
                raise ValueError(
                    f"Symptom '{symptom_id}' references cause '{cause_id}' "
                    f"which does not exist in causes.yaml"
                )


SYMPTOM_GRAPH: MappingProxyType = _build_symptom_graph()
CAUSE_GRAPH: MappingProxyType = _build_cause_graph()
_validate_cross_references(SYMPTOM_GRAPH, CAUSE_GRAPH)
