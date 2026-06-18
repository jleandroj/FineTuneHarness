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
        # Generic ML fields present in every skill
        for name in list_skills():
            spec = get_skill(name)
            assert spec is not None
            for key in ["technique", "epochs", "learning_rate", "batch_size", "max_length", "model_name"]:
                assert key in spec.input_schema
        # Biology-domain fields (k, max_per_species) are NOT in the generic core schema

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

    def test_unwired_skill_raises_not_returns_zeros(self):
        """Built-in skills must raise NotImplementedError, never silently return zeros.

        Returning zeros would pass output validation and corrupt the results CSV.
        Callers must register a real implementation via dispatcher.register() first.
        """
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
        with pytest.raises(NotImplementedError, match="no implementation registered"):
            execute_skill("lora", task)

    def test_all_unwired_skill_handlers_raise_not_return_zeros(self):
        """Every built-in skill handler raises NotImplementedError — none return fake zeros.

        Calls the handler directly (bypassing input-schema validation) to confirm
        no skill ever returns a dict of zeros that would silently corrupt the CSV.
        """
        task = TaskRecord(
            task_id="t", run_id="r", task_key="k",
            status=TaskStatus.PENDING, payload={},
        )
        for name in list_skills():
            spec = get_skill(name)
            assert spec is not None
            with pytest.raises(NotImplementedError, match="no implementation registered"):
                spec.handler(task)


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

# ── validate_common_output covers all 9 schema fields ─────────────────────────

class TestCommonInputValidator:
    """validate_common_input must distinguish missing fields from out-of-range values.

    Only generic ML fields are required; domain-specific fields (k, max_per_species)
    were extracted to skills/biology/validators.py.
    """

    _VALID = {
        "epochs": 10, "learning_rate": 1e-4, "batch_size": 32, "max_length": 512,
    }

    @pytest.fixture
    def validator(self):
        from finetuneharness.registry.builtin_skills import _common_input_validator
        return _common_input_validator

    def test_valid_payload_passes(self, validator):
        validator(dict(self._VALID))

    @pytest.mark.parametrize("field", ["epochs", "learning_rate", "batch_size", "max_length"])
    def test_missing_field_says_required(self, validator, field):
        payload = {k: v for k, v in self._VALID.items() if k != field}
        with pytest.raises(ValueError, match=f"{field} is required"):
            validator(payload)

    def test_epochs_zero_says_positive_not_missing(self, validator):
        payload = {**self._VALID, "epochs": 0}
        with pytest.raises(ValueError, match="epochs must be positive"):
            validator(payload)

    def test_error_includes_bad_value(self, validator):
        payload = {**self._VALID, "epochs": -5}
        with pytest.raises(ValueError, match="-5"):
            validator(payload)

    def test_domain_fields_not_required_by_core(self, validator):
        """k and max_per_species are biology-domain fields — the generic core must not require them."""
        validator(self._VALID)  # no k, no max_per_species → must pass


class TestBiologyInputValidator:
    """Domain-specific biology validator lives in skills/biology/validators.py."""

    _VALID_GENERIC = {
        "epochs": 10, "learning_rate": 1e-4, "batch_size": 32, "max_length": 512,
    }

    @pytest.fixture
    def bio_validator(self):
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2] / "skills"))
        from biology.validators import validate_bio_input
        return validate_bio_input

    def test_valid_bio_payload_passes(self, bio_validator):
        bio_validator({**self._VALID_GENERIC, "k": 3, "max_per_species": 100})

    def test_k_out_of_range_rejected(self, bio_validator):
        with pytest.raises(ValueError, match="k-mer"):
            bio_validator({**self._VALID_GENERIC, "k": 0})

    def test_k_too_large_rejected(self, bio_validator):
        with pytest.raises(ValueError, match="k-mer"):
            bio_validator({**self._VALID_GENERIC, "k": 7})

    def test_max_per_species_zero_rejected(self, bio_validator):
        with pytest.raises(ValueError, match="max_per_species"):
            bio_validator({**self._VALID_GENERIC, "max_per_species": 0})

    def test_bio_fields_optional(self, bio_validator):
        """Both fields are optional — a payload without them must pass."""
        bio_validator(self._VALID_GENERIC)


class TestCommonOutputValidator:
    """The common output validator must enforce all fields declared in COMMON_OUTPUT_SCHEMA.

    Before the fix, only accuracy and f1 were checked — the other 7 fields could
    hold physically impossible values (precision: -0.5, auc: 150, n_params: -1)
    and still pass validation.
    """

    @pytest.fixture
    def validator(self):
        from finetuneharness.registry.builtin_skills import _common_output_validator
        return _common_output_validator

    def test_valid_full_result_passes(self, validator):
        validator({
            "accuracy": 0.95, "f1": 0.93, "precision": 0.91, "recall": 0.94,
            "auc": 0.97, "n_params": 125_000_000, "wall_seconds": 3600.0,
            "technique": "lora", "k": 3,
        })

    def test_partial_result_passes(self, validator):
        """Fields are optional — a result with only accuracy must still pass."""
        validator({"accuracy": 0.8})

    @pytest.mark.parametrize("field", ["accuracy", "f1", "precision", "recall", "auc"])
    def test_metric_below_zero_rejected(self, validator, field):
        with pytest.raises(ValueError, match=field):
            validator({field: -0.5})

    @pytest.mark.parametrize("field", ["accuracy", "f1", "precision", "recall", "auc"])
    def test_metric_above_one_rejected(self, validator, field):
        with pytest.raises(ValueError, match=field):
            validator({field: 1.001})

    def test_n_params_negative_rejected(self, validator):
        with pytest.raises(ValueError, match="n_params"):
            validator({"n_params": -1})

    def test_n_params_zero_rejected(self, validator):
        with pytest.raises(ValueError, match="n_params"):
            validator({"n_params": 0})

    def test_wall_seconds_negative_rejected(self, validator):
        with pytest.raises(ValueError, match="wall_seconds"):
            validator({"wall_seconds": -1.0})

    def test_wall_seconds_zero_passes(self, validator):
        validator({"wall_seconds": 0.0})

    def test_technique_empty_rejected(self, validator):
        with pytest.raises(ValueError, match="technique"):
            validator({"technique": ""})

    def test_technique_whitespace_rejected(self, validator):
        with pytest.raises(ValueError, match="technique"):
            validator({"technique": "   "})

    def test_generic_validator_ignores_k(self, validator):
        """'k' is a biology-domain field — the generic core must NOT range-check it.

        A non-genomic skill may emit 'k' with a different meaning (e.g. top-k),
        so values outside the k-mer [1, 6] range must pass the generic validator.
        The k range check now lives in skills/biology/validators.py.
        """
        validator({"k": 0})    # would have raised under the old generic validator
        validator({"k": 7})
        validator({"k": 50})

    def test_audit_examples_all_rejected(self, validator):
        """Exact values from the audit report must now be caught."""
        with pytest.raises(ValueError, match="precision"):
            validator({"precision": -0.5})
        with pytest.raises(ValueError, match="auc"):
            validator({"auc": 150.0})
        with pytest.raises(ValueError, match="n_params"):
            validator({"n_params": -1})


class TestBiologyOutputValidator:
    """k-mer output range check lives in skills/biology/validators.py, not the core."""

    @pytest.fixture
    def bio_output_validator(self):
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2] / "skills"))
        from biology.validators import validate_bio_output
        return validate_bio_output

    def test_valid_k_passes(self, bio_output_validator):
        bio_output_validator({"k": 3})

    def test_k_below_range_rejected(self, bio_output_validator):
        with pytest.raises(ValueError, match="k-mer"):
            bio_output_validator({"k": 0})

    def test_k_above_range_rejected(self, bio_output_validator):
        with pytest.raises(ValueError, match="k-mer"):
            bio_output_validator({"k": 7})

    def test_k_optional(self, bio_output_validator):
        """A result without 'k' must pass."""
        bio_output_validator({"accuracy": 0.9})


# ── Registry isolation (global singleton P2) ─────────────────────────────────

class TestRegistryIsolation:
    """Two SkillRegistry instances must be independent; _GLOBAL_REGISTRY must be resettable."""

    def test_two_registries_do_not_share_state(self):
        """Registering a skill in r1 must not make it visible in r2."""
        r1 = SkillRegistry()
        r2 = SkillRegistry()

        def handler(task, **kwargs):
            return {"accuracy": 0.9, "f1": 0.89, "precision": 0.88, "recall": 0.91}

        spec = SkillSpec(
            name="isolation_test_skill",
            description="isolation test",
            input_schema={"k": int},
            output_schema={"accuracy": float, "f1": float, "precision": float, "recall": float},
            handler=handler,
        )
        r1.register(spec)
        assert r1.get("isolation_test_skill") is not None
        assert r2.get("isolation_test_skill") is None, (
            "r2 must not see r1's skills — they share underlying state"
        )

    def test_register_builtin_skills_idempotent(self):
        """Calling register_builtin_skills() twice on the same registry must not raise."""
        local = SkillRegistry()
        register_builtin_skills(local)
        # Second call must be a no-op, not a ValueError("already registered")
        register_builtin_skills(local)
        assert len(local.list_skills()) == 12

    def test_injection_populates_local_registry_not_global(self):
        """Skills registered into a local registry must not appear in the global."""
        from finetuneharness.registry import _GLOBAL_REGISTRY, _reset_global_registry

        _reset_global_registry()  # start with empty global
        local = SkillRegistry()
        register_builtin_skills(local)

        import finetuneharness.registry as reg_module
        assert len(reg_module._GLOBAL_REGISTRY.list_skills()) == 0, (
            "register_builtin_skills(local) must not mutate _GLOBAL_REGISTRY"
        )
        assert len(local.list_skills()) == 12

        # Restore global for subsequent tests
        _reset_global_registry()

    def test_reset_global_registry_gives_empty_instance(self):
        """After _reset_global_registry(), _GLOBAL_REGISTRY must be empty."""
        from finetuneharness.registry import _reset_global_registry
        import finetuneharness.registry as reg_module

        register_builtin_skills()  # ensure it has content first
        _reset_global_registry()
        assert len(reg_module._GLOBAL_REGISTRY.list_skills()) == 0, (
            "_GLOBAL_REGISTRY must be empty immediately after _reset_global_registry()"
        )
        # Restore for subsequent tests
        _reset_global_registry()

    def test_get_skill_with_local_registry(self):
        """get_skill(name, registry=local) must read from local, not global."""
        local = SkillRegistry()
        register_builtin_skills(local)
        spec = get_skill("lora", local)
        assert spec is not None
        assert spec.name == "lora"

    def test_execute_skill_with_local_registry(self):
        """execute_skill(name, task, registry=local) must dispatch through local, not global."""
        local = SkillRegistry()
        register_builtin_skills(local)
        task = TaskRecord(
            task_id="t", run_id="r", task_key="k",
            status=TaskStatus.PENDING, payload={},
        )
        # Skill exists in local — raises ValueError (input schema), which proves
        # execute_skill found "lora" in the injected registry and reached validation.
        # If dispatch had used the wrong registry, a different error path would occur.
        with pytest.raises(ValueError, match="missing required input"):
            execute_skill("lora", task, registry=local)
