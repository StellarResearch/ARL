"""Unit tests for flight telemetry time-series and benchmark figures."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

import plotly.graph_objects as go

from adaptive_rl.flight_testing.scenarios import FlightStepTelemetry
from adaptive_rl.gui.visualizer import (
    build_flight_benchmark_figure,
    build_telemetry_time_series_figure,
)


def test_build_telemetry_time_series_figure_empty() -> None:
    """Empty telemetry input produces a valid Plotly Figure without crashing."""
    fig = build_telemetry_time_series_figure([])
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text is not None


def test_build_telemetry_time_series_figure_with_data() -> None:
    """Verify trace creation across all 4 subplots from real FlightStepTelemetry."""
    telemetry = [
        FlightStepTelemetry(
            step=1,
            time_sec=0.05,
            position=np.array([5.0, 5.0, 5.0]),
            velocity=np.array([0.1, 0.2, 0.5]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            euler_angles_deg=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            target_position=np.array([10.0, 10.0, 5.0]),
            distance_to_target=7.07,
            action=np.zeros(4, dtype=np.float32),
            motor_thrusts=np.full(4, 2.45),
            min_obstacle_distance=3.5,
            is_collision=False,
            is_success=False,
        ),
        FlightStepTelemetry(
            step=2,
            time_sec=0.10,
            position=np.array([5.01, 5.02, 5.03]),
            velocity=np.array([0.15, 0.22, 0.48]),
            quaternion=np.array([0.999, 0.01, 0.0, 0.0]),
            euler_angles_deg=np.array([1.1, 0.0, 0.0]),
            angular_velocity=np.array([0.05, 0.0, 0.0]),
            target_position=np.array([10.0, 10.0, 5.0]),
            distance_to_target=7.04,
            action=np.zeros(4, dtype=np.float32),
            motor_thrusts=np.full(4, 2.46),
            min_obstacle_distance=3.48,
            is_collision=False,
            is_success=False,
        ),
    ]

    fig = build_telemetry_time_series_figure(telemetry, title="Test Flight Telemetry")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 8  # Multiple traces across the 4 panels
    trace_names = [tr.name for tr in fig.data]
    assert "Altitude (Z)" in trace_names
    assert "Speed ||v||" in trace_names
    assert "Roll φ (°)" in trace_names
    assert "Goal Dist (m)" in trace_names


def test_build_flight_benchmark_figure() -> None:
    """Verify grouped bar chart creation from benchmark combination results."""
    combos = [
        {
            "algorithm": "PPO",
            "mode": "3-DOF",
            "success_rate": 0.85,
            "collision_rate": 0.10,
            "timeout_rate": 0.05,
        },
        {
            "algorithm": "PPO",
            "mode": "6-DOF",
            "success_rate": 0.70,
            "collision_rate": 0.20,
            "timeout_rate": 0.10,
        },
        {
            "algorithm": "SAC",
            "mode": "3-DOF",
            "success_rate": 0.90,
            "collision_rate": 0.05,
            "timeout_rate": 0.05,
        },
        {
            "algorithm": "SAC",
            "mode": "6-DOF",
            "success_rate": 0.75,
            "collision_rate": 0.15,
            "timeout_rate": 0.10,
        },
    ]

    fig = build_flight_benchmark_figure(combos, title="Benchmark Test")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # Success Rate, Collision Rate, Timeout Rate bars
    assert fig.data[0].name == "Success Rate (%)"
    assert fig.data[1].name == "Collision Rate (%)"
    assert fig.data[2].name == "Timeout Rate (%)"
    assert len(fig.data[0].x) == 4
