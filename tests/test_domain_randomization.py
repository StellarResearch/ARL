"""Unit tests for domain randomization, parameter shifts, and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adaptive_rl.environments.drone_6dof import DroneDynamics6DOF
from adaptive_rl.environments.randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
)


def test_domain_randomization_bounds_and_sampling() -> None:
    """Sampled parameters must strictly respect configured min/max scale intervals."""
    cfg = DomainRandomizationConfig(
        mass_scale=(0.9, 1.1),
        max_thrust_scale=(0.95, 1.05),
        wind_speed_max=3.0,
    )
    randomizer = DomainRandomizer(config=cfg, seed=42)
    params = randomizer.sample_parameters()

    assert 0.9 <= params["mass"] <= 1.1
    assert 0.95 * 5.0 <= params["max_thrust_per_motor"] <= 1.05 * 5.0
    wind_mag = float(np.linalg.norm(params["wind_vector"]))
    assert wind_mag <= 3.0001


def test_domain_randomization_reproducibility() -> None:
    """Identical seeds yield identical parameter sets."""
    r1 = DomainRandomizer(seed=999)
    r2 = DomainRandomizer(seed=999)

    p1 = r1.sample_parameters(seed=123)
    p2 = r2.sample_parameters(seed=123)

    assert p1 == p2


def test_domain_randomization_applied_to_dynamics() -> None:
    """Applying sampled parameters updates physical properties on active dynamics model."""
    dyn = DroneDynamics6DOF(mass=1.0)
    randomizer = DomainRandomizer()

    custom_params = {
        "mass": 1.25,
        "inertia": [0.015, 0.015, 0.030],
        "max_thrust_per_motor": 6.5,
        "wind_vector": [2.0, -1.0, 0.5],
        "motor_time_constant": 0.045,
    }
    randomizer.apply_to_dynamics(dyn, custom_params)

    assert dyn.mass == 1.25
    np.testing.assert_allclose(dyn.inertia_diag, [0.015, 0.015, 0.030])
    assert dyn.max_thrust_per_motor == 6.5
    np.testing.assert_allclose(dyn.wind_vector, [2.0, -1.0, 0.5])
    assert dyn.motor_time_constant == 0.045
    # Recalculates equilibrium hover thrust per motor
    assert dyn.hover_thrust_per_motor == (1.25 * 9.81) / 4.0


def test_domain_randomization_config_export(tmp_path: Path) -> None:
    """Configuration exports to valid JSON and reloads."""
    r = DomainRandomizer()
    out = r.export_config(tmp_path / "dr_config.json")
    assert out.is_file()

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "mass_scale" in data
    assert "wind_speed_max" in data
