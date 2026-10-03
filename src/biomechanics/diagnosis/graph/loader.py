"""YAML graph loader — loads a symptom/cause YAML pair into frozen dicts.

Resolves string function references to actual callables from an evidence-test
module and a parameter-delta module. Validates all cross-references at load
time. load_graph builds any exercise's pair; the squat's is built at import.

Exports:
    load_graph    — build (symptom graph, cause graph) from a YAML pair
    SYMPTOM_GRAPH — MappingProxyType of the squat's symptom definitions
    CAUSE_GRAPH   — MappingProxyType of the squat's cause definitions
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType, ModuleType
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


def _build_symptom_graph(filename: str, evidence_module: ModuleType) -> MappingProxyType:
    raw = _load_yaml(filename)
    graph: dict[str, dict[str, Any]] = {}

    for symptom_id, definition in raw.items():
        expected_fn_name = definition["expected_value_fn"]
        expected_fn = _resolve_function(evidence_module, expected_fn_name)

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


def _build_cause_graph(
    filename: str, evidence_module: ModuleType, delta_module: ModuleType
) -> MappingProxyType:
    raw = _load_yaml(filename)
    graph: dict[str, dict[str, Any]] = {}

    for cause_id, definition in raw.items():
        evidence_fn = _resolve_function(
            evidence_module, definition["evidence_test_fn"]
        )

        delta_fn_name = definition.get("parameter_delta_fn")
        delta_fn = None
        if delta_fn_name:
            delta_fn = _resolve_function(delta_module, delta_fn_name)

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
    symptom_graph: MappingProxyType, cause_graph: MappingProxyType, causes_filename: str
) -> None:
    for symptom_id, symptom_def in symptom_graph.items():
        for candidate in symptom_def["candidate_causes"]:
            cause_id = candidate["cause_id"]
            if cause_id not in cause_graph:
                raise ValueError(
                    f"Symptom '{symptom_id}' references cause '{cause_id}' "
                    f"which does not exist in {causes_filename}"
                )


def load_graph(
    symptoms_filename: str,
    causes_filename: str,
    evidence_module: ModuleType,
    delta_module: ModuleType,
) -> tuple[MappingProxyType, MappingProxyType]:
    """Symptom and cause graphs from a YAML pair in this directory, with every
    function name resolved in the given modules and every cause reference checked."""
    symptom_graph = _build_symptom_graph(symptoms_filename, evidence_module)
    cause_graph = _build_cause_graph(causes_filename, evidence_module, delta_module)
    _validate_cross_references(symptom_graph, cause_graph, causes_filename)
    return symptom_graph, cause_graph


SYMPTOM_GRAPH, CAUSE_GRAPH = load_graph(
    "symptoms.yaml", "causes.yaml", evidence_tests, parameter_deltas
)
