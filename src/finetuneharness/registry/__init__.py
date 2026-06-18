"""FineTuneHarness skill registry.

Defines SkillSpec contract and provides a registry for fine-tuning techniques.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "SkillSpec",
    "SkillRegistry",
    "register_builtin_skills",
    "get_skill",
    # Task dispatch
    "TaskDispatcher",
    "validate_task_payload",
    # Hooks
    "GPUMemoryHook",
    "CheckpointHook",
    "MetricsHook",
    "EarlyStoppingHook",
    "CleanupHook",
    "ProgressHook",
    "register_default_hooks",
]


@dataclass(frozen=True)
class SkillSpec:
    """Contract for a fine-tuning skill.

    Args:
        name: Unique skill identifier (e.g., "lora", "sft")
        description: Human-readable description
        input_schema: Dict defining required input keys and types
        output_schema: Dict defining required output keys and types
        handler: Callable that implements the skill. Receives TaskRecord, returns result dict.
        validate_input: Optional custom input validator
        validate_output: Optional custom output validator
    """

    name: str
    description: str
    input_schema: dict[str, type]
    output_schema: dict[str, type]
    handler: Callable[..., dict[str, Any]]
    validate_input: Callable[[dict[str, Any]], None] | None = None
    validate_output: Callable[[dict[str, Any]], None] | None = None


class SkillRegistry:
    """Registry for fine-tuning skills with input/output validation."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        """Register a skill. Raises ValueError if name already exists."""
        if spec.name in self._skills:
            raise ValueError(f"Skill '{spec.name}' already registered")
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec | None:
        """Get skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """Return sorted list of registered skill names."""
        return sorted(self._skills.keys())

    def validate_input(self, skill_name: str, payload: dict[str, Any]) -> None:
        """Validate payload against skill's input schema."""
        spec = self._skills.get(skill_name)
        if spec is None:
            raise KeyError(f"Unknown skill: {skill_name}")

        # Check required keys
        for key, expected_type in spec.input_schema.items():
            if key not in payload:
                raise ValueError(f"Skill '{skill_name}' missing required input: {key}")
            if not isinstance(payload[key], expected_type):
                raise TypeError(
                    f"Skill '{skill_name}' input '{key}' must be {expected_type.__name__}, "
                    f"got {type(payload[key]).__name__}"
                )

        # Run custom validator if provided
        if spec.validate_input:
            spec.validate_input(payload)

    def validate_output(self, skill_name: str, result: dict[str, Any]) -> None:
        """Validate result against skill's output schema."""
        spec = self._skills.get(skill_name)
        if spec is None:
            raise KeyError(f"Unknown skill: {skill_name}")

        for key, expected_type in spec.output_schema.items():
            if key not in result:
                raise ValueError(f"Skill '{skill_name}' missing required output: {key}")
            if not isinstance(result[key], expected_type):
                raise TypeError(
                    f"Skill '{skill_name}' output '{key}' must be {expected_type.__name__}, "
                    f"got {type(result[key]).__name__}"
                )

        if spec.validate_output:
            spec.validate_output(result)

    def execute(self, skill_name: str, task_record, **kwargs) -> dict[str, Any]:
        """Execute a skill with validation."""
        spec = self._skills.get(skill_name)
        if spec is None:
            raise KeyError(f"Unknown skill: {skill_name}")

        payload = dict(task_record.payload)
        payload.update(kwargs)
        self.validate_input(skill_name, payload)

        result = spec.handler(task_record, **payload)
        self.validate_output(skill_name, result)
        return result


# Global registry instance
_registry = SkillRegistry()


def register_builtin_skills() -> None:
    """Register the 12 built-in fine-tuning techniques from promoter_species_id."""
    from finetuneharness.registry.builtin_skills import BUILTIN_SKILLS

    for spec in BUILTIN_SKILLS:
        _registry.register(spec)


def get_skill(name: str) -> SkillSpec | None:
    """Get a built-in skill by name."""
    if not _registry.list_skills():
        register_builtin_skills()
    return _registry.get(name)


def list_skills() -> list[str]:
    """List all registered skills."""
    if not _registry.list_skills():
        register_builtin_skills()
    return _registry.list_skills()


def execute_skill(skill_name: str, task_record, **kwargs) -> dict[str, Any]:
    """Execute a skill with validation."""
    if not _registry.list_skills():
        register_builtin_skills()
    return _registry.execute(skill_name, task_record, **kwargs)


from finetuneharness.registry.dispatcher import TaskDispatcher, validate_task_payload

# Import and expose hooks
from finetuneharness.registry.hooks import (
    GPUMemoryHook,
    CheckpointHook,
    MetricsHook,
    EarlyStoppingHook,
    CleanupHook,
    ProgressHook,
    register_default_hooks,
)