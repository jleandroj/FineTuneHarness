"""Built-in fine-tuning skills for FineTuneHarness.

Generic SkillSpec definitions for 12 fine-tuning techniques. Each spec declares
the input/output schema and a stub handler; the real implementation is injected
at runtime via TaskDispatcher.register().

Techniques:
- sft: Full fine-tuning
- lora: LoRA (Low-Rank Adaptation)
- adalora: AdaLoRA (adaptive rank allocation)
- ia3: IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- prefix: Prefix Tuning
- prompt: Prompt Tuning
- adapter: Houlsby Adapter
- bitfit: BitFit (bias-only fine-tuning)
- curriculum: Curriculum Learning
- merging: Model Merging (SLERP, TIES, DARE)
- ewc: Elastic Weight Consolidation (continual learning)
- distil: Knowledge Distillation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from finetuneharness.registry import SkillSpec


# Common input schema for all fine-tuning skills
COMMON_INPUT_SCHEMA: dict[str, type] = {
    "technique": str,
    "epochs": int,
    "learning_rate": float,
    "batch_size": int,
    "max_length": int,
    "model_name": str,
}

# Common output schema for all fine-tuning skills
COMMON_OUTPUT_SCHEMA: dict[str, type] = {
    "accuracy": float,
    "f1": float,
    "precision": float,
    "recall": float,
    "auc": float,
    "n_params": int,
    "wall_seconds": float,
    "technique": str,
}


def _make_common_validators():
    """Create common input/output validators."""

    def validate_common_input(payload: dict[str, Any]) -> None:
        required = ("epochs", "learning_rate", "batch_size", "max_length")
        for field in required:
            if field not in payload:
                raise ValueError(f"{field} is required")
        if payload["epochs"] <= 0:
            raise ValueError(f"epochs must be positive, got {payload['epochs']}")
        if payload["learning_rate"] <= 0:
            raise ValueError(f"learning_rate must be positive, got {payload['learning_rate']}")
        if payload["batch_size"] <= 0:
            raise ValueError(f"batch_size must be positive, got {payload['batch_size']}")
        if payload["max_length"] <= 0:
            raise ValueError(f"max_length must be positive, got {payload['max_length']}")

    def validate_common_output(result: dict[str, Any]) -> None:
        for field in ("accuracy", "f1", "precision", "recall", "auc"):
            if field in result:
                v = result[field]
                if not (0 <= v <= 1):
                    raise ValueError(f"{field} must be in [0, 1], got {v}")
        if "n_params" in result:
            if not (isinstance(result["n_params"], int) and result["n_params"] > 0):
                raise ValueError(f"n_params must be a positive int, got {result['n_params']}")
        if "wall_seconds" in result:
            if result["wall_seconds"] < 0:
                raise ValueError(f"wall_seconds must be >= 0, got {result['wall_seconds']}")
        if "technique" in result:
            if not isinstance(result["technique"], str) or not result["technique"].strip():
                raise ValueError(f"technique must be a non-empty string, got {result['technique']!r}")
        # Note: 'k' (k-mer size) is a biology-domain field — its range check lives
        # in skills/biology/validators.py:validate_bio_output, not in the generic core.

    return validate_common_input, validate_common_output


_common_input_validator, _common_output_validator = _make_common_validators()


def _create_skill_handler(technique: str) -> Callable:
    """Return a handler that raises NotImplementedError at call time.

    Built-in SkillSpecs define the *contract* (input/output schema, validators).
    The *implementation* must be registered via TaskDispatcher before running:

        dispatcher.register("lora", my_lora_training_fn)

    Returning zeros here would be silent data corruption — the validator would
    accept them as valid results and they would be written to the CSV.
    """

    def handler(task_record, **payload) -> dict[str, Any]:
        raise NotImplementedError(
            f"Skill '{technique}' has no implementation registered. "
            f"Call dispatcher.register('{technique}', your_handler) "
            f"before running tasks of this kind. "
            f"See finetuneharness/registry/builtin_skills.py for the expected output schema."
        )

    handler.__name__ = f"run_{technique}"
    return handler


# Define all 12 skills
BUILTIN_SKILLS: list[SkillSpec] = [
    SkillSpec(
        name="sft",
        description="Full fine-tuning (Supervised Fine-Tuning) - updates all model parameters",
        input_schema=COMMON_INPUT_SCHEMA,
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("sft"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="lora",
        description="LoRA (Low-Rank Adaptation) - injects trainable low-rank matrices",
        input_schema={**COMMON_INPUT_SCHEMA, "lora_rank": int, "lora_alpha": int, "lora_dropout": float},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("lora"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="adalora",
        description="AdaLoRA - adaptive rank allocation during training",
        input_schema={**COMMON_INPUT_SCHEMA, "lora_rank": int, "target_rank": int, "lora_alpha": int},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("adalora"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="ia3",
        description="IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)",
        input_schema={**COMMON_INPUT_SCHEMA, "ia3_scale": float},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("ia3"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="prefix",
        description="Prefix Tuning - prepends trainable prefix tokens to each layer",
        input_schema={**COMMON_INPUT_SCHEMA, "num_virtual_tokens": int, "prefix_projection": bool},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("prefix"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="prompt",
        description="Prompt Tuning - prepends trainable prompt tokens to input only",
        input_schema={**COMMON_INPUT_SCHEMA, "num_virtual_tokens": int},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("prompt"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="adapter",
        description="Houlsby Adapter - bottleneck adapter layers between transformer blocks",
        input_schema={**COMMON_INPUT_SCHEMA, "adapter_size": int, "adapter_dropout": float},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("adapter"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="bitfit",
        description="BitFit - only fine-tunes bias parameters",
        input_schema=COMMON_INPUT_SCHEMA,
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("bitfit"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="curriculum",
        description="Curriculum Learning - trains on easy examples first, then harder",
        input_schema={**COMMON_INPUT_SCHEMA, "curriculum_epochs": int, "difficulty_schedule": str},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("curriculum"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="merging",
        description="Model Merging - combines multiple fine-tuned models (SLERP, TIES, DARE)",
        input_schema={**COMMON_INPUT_SCHEMA, "merge_method": str, "merge_weights": list},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("merging"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="ewc",
        description="Elastic Weight Consolidation - continual learning with Fisher information",
        input_schema={**COMMON_INPUT_SCHEMA, "ewc_lambda": float, "fisher_samples": int},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("ewc"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
    SkillSpec(
        name="distil",
        description="Knowledge Distillation - teacher-student training with soft targets",
        input_schema={**COMMON_INPUT_SCHEMA, "teacher_model": str, "temperature": float, "alpha": float},
        output_schema=COMMON_OUTPUT_SCHEMA,
        handler=_create_skill_handler("distil"),
        validate_input=_common_input_validator,
        validate_output=_common_output_validator,
    ),
]


def get_skill_spec(name: str) -> SkillSpec | None:
    """Get a built-in skill spec by name without registering all."""
    for spec in BUILTIN_SKILLS:
        if spec.name == name:
            return spec
    return None