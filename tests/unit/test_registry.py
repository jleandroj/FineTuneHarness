"""Tests for the skill registry and built-in skills."""

from __future__ import annotations

import pytest

from finetuneharness.registry import (
    SkillRegistry,
    SkillSpec,
    execute_skill,
    get_skill,
    list_skills,
    register_builtin_skills,
)
from finetuneharness.state.models import TaskRecord, TaskStatus


class TestSkillRegistry:
    def test_register_and_get_skill(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int, "epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)

        assert registry.get("test_skill") == spec
        assert registry.get("nonexistent") is None

    def test_register_duplicate_raises(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int, "epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_list_skills(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        for name in ["a", "b", "c"]:
            spec = SkillSpec(
                name=name,
                description=f"Skill {name}",
                input_schema={"k": int},
                output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
                handler=handler,
            )
            registry.register(spec)

        assert registry.list_skills() == ["a", "b", "c"]

    def test_validate_input_success(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int, "epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)
        registry.validate_input("test_skill", {"k": 3, "epochs": 10})

    def test_validate_input_missing_key(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int, "epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)
        with pytest.raises(ValueError, match="missing required input"):
            registry.validate_input("test_skill", {"k": 3})

    def test_validate_input_wrong_type(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int, "epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)
        with pytest.raises(TypeError, match="must be int"):
            registry.validate_input("test_skill", {"k": "3", "epochs": 10})

    def test_validate_output_success(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)
        registry.validate_output("test_skill", {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91})

    def test_validate_output_missing_key(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)
        with pytest.raises(ValueError, match="missing required output"):
            registry.validate_output("test_skill", {"accuracy": 0.9, "f1": 0.89})

    def test_execute_skill(self):
        registry = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"k": int, "epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        registry.register(spec)

        task = TaskRecord(
            task_id="test-1",
            run_id="run-1",
            task_key="test",
            status=TaskStatus.PENDING,
            payload={"k": 3, "epochs": 10},
        )
        result = registry.execute("test_skill", task)
        assert result["accuracy"] == 0.9
        assert result["f1"] == 0.89


class TestBuiltinSkills:
    @classmethod
    def setup_class(cls):
        register_builtin_skills()

    def test_all_12_skills_registered(self):
        skills = list_skills()
        expected = sorted([
            "sft",
            "lora",
            "adalora",
            "ia3",
            "prefix",
            "prompt",
            "adapter",
            "bitfit",
            "curriculum",
            "merging",
            "ewc",
            "distil",
        ])
        assert skills == expected

    def test_each_skill_has_valid_spec(self):
        for name in list_skills():
            spec = get_skill(name)
            assert spec is not None
            assert spec.name == name
            assert spec.description
            assert spec.input_schema
            assert spec.output_schema
            assert spec.handler is not None

    def test_skill_input_schemas(self):
        # Common fields
        for name in list_skills():
            spec = get_skill(name)
            assert spec is not None
            for key in ["k", "technique", "epochs", "max_per_species", "learning_rate", "batch_size", "max_length", "model_name"]:
                assert key in spec.input_schema

        # Technique-specific fields
        lora_spec = get_skill("lora")
        assert lora_spec is not None
        assert "lora_rank" in lora_spec.input_schema
        assert "lora_alpha" in lora_spec.input_schema
        assert "lora_dropout" in lora_spec.input_schema

        adalora_spec = get_skill("adalora")
        assert adalora_spec is not None
        assert "target_rank" in adalora_spec.input_schema

        ia3_spec = get_skill("ia3")
        assert ia3_spec is not None
        assert "ia3_scale" in ia3_spec.input_schema

    def test_execute_builtin_skill(self):

        task = TaskRecord(
            task_id="test-1",
            run_id="run-1",
            task_key="k3-lora",
            status=TaskStatus.PENDING,
            payload={
                "k": 3,
                "technique": "lora",
                "epochs": 5,
                "max_per_species": 1000,
                "learning_rate": 2e-4,
                "batch_size": 16,
                "max_length": 512,
                "model_name": "bert-base-uncased",
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
            },
        )

        result = execute_skill("lora", task)
        assert "accuracy" in result
        assert "f1" in result
        assert "wall_seconds" in result
        assert result["technique"] == "lora"
        assert result["k"] == 3


class TestCustomValidators:
    def test_custom_input_validator(self):
        registry = SkillRegistry()

        def validator(payload):
            if payload.get("epochs", 0) > 100:
                raise ValueError("epochs too high")

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
            validate_input=validator,
        )
        registry.register(spec)

        registry.validate_input("test_skill", {"epochs": 10})  # OK
        with pytest.raises(ValueError, match="epochs too high"):
            registry.validate_input("test_skill", {"epochs": 200})

    def test_custom_output_validator(self):
        registry = SkillRegistry()

        def validator(result):
            if result.get("accuracy", 0) < 0.5:
                raise ValueError("accuracy too low")

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="test_skill",
            description="Test skill",
            input_schema={"epochs": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
            validate_output=validator,
        )
        registry.register(spec)

        registry.validate_output("test_skill", {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91})
        with pytest.raises(ValueError, match="accuracy too low"):
            registry.validate_output("test_skill", {"accuracy": 0.3, "f1": 0.4, "precision": 0.5, "recall": 0.6})