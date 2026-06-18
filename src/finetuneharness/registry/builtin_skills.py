"""Built-in fine-tuning skills for FineTuneHarness.

These 12 techniques correspond to the ablation grid in promoter_species_id:
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
    "k": int,  # k-mer size (1-6)
    "technique": str,  # technique name
    "epochs": int,
    "max_per_species": int,
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
    "k": int,
}


def _make_common_validators():
    """Create common input/output validators."""

    def validate_common_input(payload: dict[str, Any]) -> None:
        if not (1 <= payload.get("k", 0) <= 6):
            raise ValueError("k must be between 1 and 6")
        if payload.get("epochs", 0) <= 0:
            raise ValueError("epochs must be positive")
        if payload.get("max_per_species", 0) <= 0:
            raise ValueError("max_per_species must be positive")
        if payload.get("learning_rate", 0) <= 0:
            raise ValueError("learning_rate must be positive")
        if payload.get("batch_size", 0) <= 0:
            raise ValueError("batch_size must be positive")
        if payload.get("max_length", 0) <= 0:
            raise ValueError("max_length must be positive")

    def validate_common_output(result: dict[str, Any]) -> None:
        acc = result.get("accuracy", -1)
        if not (0 <= acc <= 1):
            raise ValueError(f"accuracy must be in [0, 1], got {acc}")
        f1 = result.get("f1", -1)
        if not (0 <= f1 <= 1):
            raise ValueError(f"f1 must be in [0, 1], got {f1}")

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