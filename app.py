"""AdaptiveRL: Autonomous 3D Drone Navigation Flight Deck GUI.

A premium aerospace-grade mission-control Streamlit interface for evaluating
autonomous drone flight systems, rigid-body 6-DOF dynamics, 3-DOF kinematic simulation,
16-ray spherical LiDAR sensing, PPO training, and empirical policy benchmarking.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import streamlit as st

from adaptive_rl.algorithms.registry import (
    inspect_model_checkpoint,
    load_algorithm_from_pretrained,
)
from adaptive_rl.benchmarking import FlightBenchmarkSuite
from adaptive_rl.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig, TrainingConfig
from adaptive_rl.controllers import QuadrotorPIDController
from adaptive_rl.environments.drone import (
    DroneNavigation3DEnv,
    validate_target_coordinates,
)
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv, quaternion_to_euler_angles
from adaptive_rl.evaluation.evaluator import (
    Evaluator,
    evaluate_policy,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.flight_testing import FlightScenarioRunner
from adaptive_rl.gui import (
    build_3d_navigation_trajectory_figure,
    build_actuator_telemetry_figure,
    build_arena_3d_figure,
    build_comparison_bar_chart,
    build_density_experiment_figure,
    build_flight_benchmark_figure,
    build_telemetry_time_series_figure,
    build_training_curve_figure,
    get_available_models,
    run_drone_simulation_episode,
)
from adaptive_rl.training import RLTrainer, TrainingProgressCallback

# Configure Streamlit Page
st.set_page_config(
    page_title="ARL | Autonomous Flight Systems — 3D Flight Deck",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_custom_aerospace_css() -> None:
    """Inject premium dark aerospace mission-control CSS styling."""
    st.markdown(
        """
        <style>
        :root {
            --arl-bg: #080c14;
            --arl-panel: #0f172a;
            --arl-panel-border: #1e293b;
            --arl-cyan: #00f2fe;
            --arl-blue: #38bdf8;
            --arl-accent: #2563eb;
            --arl-green: #10b981;
            --arl-amber: #f59e0b;
            --arl-red: #ef4444;
            --arl-text: #f1f5f9;
            --arl-muted: #94a3b8;
        }

        /* Mission Control Top Header */
        .arl-header-card {
            background: linear-gradient(135deg, #091326 0%, #0d1b38 60%, #11224d 100%);
            border: 1px solid #1e3a66;
            border-radius: 8px;
            padding: 16px 22px;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
        }

        .arl-brand-title {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: 2.2px;
            color: #ffffff;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .arl-brand-title span.accent {
            color: #00f2fe;
            text-shadow: 0 0 12px rgba(0, 242, 254, 0.45);
        }

        .arl-brand-sub {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.5px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-top: 3px;
            font-family: "JetBrains Mono", "SF Mono", monospace;
        }

        .arl-badge-group {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }

        .arl-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            font-family: "JetBrains Mono", "SF Mono", monospace;
        }

        .arl-pill-cyan {
            background: rgba(0, 242, 254, 0.12);
            border: 1px solid rgba(0, 242, 254, 0.4);
            color: #00f2fe;
        }

        .arl-pill-blue {
            background: rgba(37, 99, 235, 0.18);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #38bdf8;
        }

        .arl-pill-green {
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #10b981;
        }

        .arl-pill-amber {
            background: rgba(245, 158, 11, 0.14);
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: #f59e0b;
        }

        .arl-pill-red {
            background: rgba(239, 68, 68, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #ef4444;
        }

        /* Mission Control Toolbar */
        .arl-toolbar {
            background: #0d1527;
            border: 1px solid #1e2e4a;
            border-radius: 8px;
            padding: 14px 18px 10px 18px;
            margin-bottom: 14px;
        }

        /* Flight Outcome Banner */
        .arl-outcome-banner {
            border-radius: 6px;
            padding: 12px 18px;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .arl-outcome-banner.success {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.45);
            border-left: 5px solid #10b981;
        }

        .arl-outcome-banner.failure {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.45);
            border-left: 5px solid #ef4444;
        }

        .arl-outcome-banner.timeout {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.45);
            border-left: 5px solid #f59e0b;
        }

        /* Telemetry Cards */
        .telemetry-card {
            background: #0d1527;
            border: 1px solid #1e304f;
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }

        .telemetry-header {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #38bdf8;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            border-bottom: 1px solid rgba(56, 189, 248, 0.2);
            padding-bottom: 4px;
            font-family: "JetBrains Mono", "SF Mono", monospace;
        }

        .telemetry-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px 12px;
        }

        .telemetry-stat {
            display: flex;
            flex-direction: column;
        }

        .telemetry-stat-label {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            color: #94a3b8;
            letter-spacing: 0.5px;
        }

        .telemetry-stat-val {
            font-size: 16px;
            font-weight: 700;
            font-family: "JetBrains Mono", "SF Mono", monospace;
            color: #f8fafc;
        }

        .telemetry-stat-val.cyan { color: #00f2fe; }
        .telemetry-stat-val.green { color: #10b981; }
        .telemetry-stat-val.amber { color: #f59e0b; }
        .telemetry-stat-val.red { color: #ef4444; }

        .telemetry-stat-val span.unit {
            font-size: 11px;
            font-weight: normal;
            color: #64748b;
            margin-left: 2px;
        }

        /* Subtle disclaimer */
        .baseline-disclaimer {
            font-size: 11px;
            color: #94a3b8;
            background: #111b33;
            border: 1px solid #1e355b;
            border-radius: 4px;
            padding: 6px 10px;
            margin-top: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_aerospace_css()

    # Sidebar: Project Overview & Identity
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1e3a66; margin-bottom: 14px;">
            <div style="font-size: 18px; font-weight: 800; letter-spacing: 1.5px; color: #ffffff;">
                🛸 <span style="color: #00f2fe;">ARL</span> FLIGHT SYSTEMS
            </div>
            <div style="font-size: 10px; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase;">
                Autonomous Navigation Research Deck
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "**Aerospace Simulation Platform**\n\n"
        "Unified evaluation deck for 6-DOF rigid-body dynamics (RK4 integration) and "
        "3-DOF kinematic drone simulation with deep reinforcement learning (PPO & SAC).\n\n"
        "---\n"
        "**Key Capabilities**\n"
        "- 🛰️ 3D Mesh Airframe Telemetry\n"
        "- 📡 16-Ray Spherical LiDAR Array\n"
        "- 🔄 Playback Numerical Scrubber\n"
        "- 🧪 Obstacle-Density Benchmarks"
    )

    available_models = get_available_models()
    model_info_map = {m.name: inspect_model_checkpoint(m) for m in available_models}
    model_names = [m.name for m in available_models]

    tabs = st.tabs(
        [
            "🎮 Live 3D Flight Deck",
            "📈 Train Policy",
            "📊 Evaluation & Baseline",
            "🎯 Difficulty Experiment",
            "📘 About & Architecture",
            "🚀 Flight Scenarios & Benchmarks",
        ]
    )

    # ==========================================
    # TAB 1: LIVE 3D FLIGHT DECK
    # ==========================================
    with tabs[0]:
        # Professional ARL Mission Header
        st.markdown(
            """
            <div class="arl-header-card">
                <div class="arl-brand">
                    <div class="arl-brand-title">
                        ARL <span class="accent">|</span> AUTONOMOUS FLIGHT SYSTEMS
                    </div>
                    <div class="arl-brand-sub">
                        HIGH-FIDELITY FLIGHT DYNAMICS & NEURAL NAVIGATION DECK
                    </div>
                </div>
                <div class="arl-badge-group">
                    <div class="arl-pill arl-pill-cyan">● SYSTEM ACTIVE</div>
                    <div class="arl-pill arl-pill-blue">SIMULATION PLAYBACK</div>
                    <div class="arl-pill arl-pill-green">16-RAY LIDAR READY</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Visualizing simulated quadrotor trajectory, rigid-body quaternion attitude, and spherical LiDAR "
            "rangefinder readings. Recorded step playback from numerical physics integration (RK4/Euler)."
        )

        # Simulation Control Toolbar / Card
        with st.container():
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns(
                [1.3, 1.8, 0.9, 0.9, 1.0, 1.2]
            )

            with ctrl_col1:
                env_choice = st.selectbox(
                    "Dynamics Model:",
                    options=["6-DOF Quadrotor", "3-DOF Kinematic Drone"],
                    index=0,
                    help="6-DOF rigid-body dynamics (quaternion attitude & 4 rotor thrusts) or 3-DOF point-mass kinematics.",
                )

            with ctrl_col2:
                policy_options = ["Random Action Baseline"]
                for m in available_models:
                    info = model_info_map[m.name]
                    if info["valid"] and info["algorithm"]:
                        algo_label = info["algorithm"].upper()
                        env_label = f" ({info['compatible_env']})" if info["compatible_env"] else ""
                        policy_options.append(f"Trained {algo_label}{env_label}: {m.name}")
                    else:
                        policy_options.append(f"Unsupported: {m.name}")

                selected_policy = st.selectbox(
                    "Select Policy / Checkpoint:",
                    options=policy_options,
                    index=0,
                    help="Choose a trained neural network checkpoint or test an untrained random baseline.",
                )

            with ctrl_col3:
                demo_seed = st.number_input(
                    "Environment Seed:",
                    min_value=0,
                    max_value=999999,
                    value=42,
                    step=1,
                    help="Locks procedural obstacle placement and target location for reproducible evaluation.",
                )

            with ctrl_col4:
                num_obs = st.slider(
                    "Obstacle Count:",
                    min_value=0,
                    max_value=8,
                    value=4,
                    step=1,
                    help="Number of procedural spherical obstacles generated in the arena.",
                )

            with ctrl_col5:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                show_lidar_rays = st.checkbox("16-Ray LiDAR", value=True)

            with ctrl_col6:
                st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
                launch_clicked = st.button("🚀 Execute Sim", type="primary", width="stretch")

            if selected_policy.startswith("Random Action Baseline"):
                st.markdown(
                    "<div class='baseline-disclaimer'>⚠️ <b>Untrained Baseline Note:</b> Random Action Baseline provides "
                    "stochastic control inputs for baseline comparison; it does not represent a trained agent.</div>",
                    unsafe_allow_html=True,
                )

        is_6dof = "6-DOF" in env_choice
        active_bounds = (30.0, 30.0, 15.0)
        default_start = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        default_goal = np.array(
            [active_bounds[0] - 5.0, active_bounds[1] - 5.0, active_bounds[2] - 5.0],
            dtype=np.float64,
        )
        target_radius_val = 1.5
        collision_radius_val = 0.8

        # ====================================================
        # CUSTOM DESTINATION / TARGET LOCATION SECTION
        # ====================================================
        st.markdown(
            """
            <div style="margin-top: 12px; margin-bottom: 6px; font-weight: 700; font-size: 13px; letter-spacing: 1.2px; text-transform: uppercase; color: #38bdf8; display: flex; align-items: center; gap: 8px;">
                🎯 CUSTOM DESTINATION / TARGET LOCATION
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            dest_col1, dest_col2, dest_col3, dest_col4 = st.columns([1.1, 1.1, 1.1, 1.5])

            with dest_col1:
                input_tx = st.number_input(
                    "Target X (m):",
                    value=float(st.session_state.get("target_x", default_goal[0])),
                    min_value=-15.0,
                    max_value=float(active_bounds[0] + 15.0),
                    step=1.0,
                    format="%.2f",
                    key="custom_target_x",
                    help=f"Desired X coordinate in metres (valid range: [{collision_radius_val:.1f}, {active_bounds[0] - collision_radius_val:.1f}] m).",
                )
                st.session_state["target_x"] = float(input_tx)

            with dest_col2:
                input_ty = st.number_input(
                    "Target Y (m):",
                    value=float(st.session_state.get("target_y", default_goal[1])),
                    min_value=-15.0,
                    max_value=float(active_bounds[1] + 15.0),
                    step=1.0,
                    format="%.2f",
                    key="custom_target_y",
                    help=f"Desired Y coordinate in metres (valid range: [{collision_radius_val:.1f}, {active_bounds[1] - collision_radius_val:.1f}] m).",
                )
                st.session_state["target_y"] = float(input_ty)

            with dest_col3:
                input_tz = st.number_input(
                    "Target Z (m):",
                    value=float(st.session_state.get("target_z", default_goal[2])),
                    min_value=-15.0,
                    max_value=float(active_bounds[2] + 15.0),
                    step=1.0,
                    format="%.2f",
                    key="custom_target_z",
                    help=f"Desired Z coordinate in metres (valid range: [{collision_radius_val:.1f}, {active_bounds[2] - collision_radius_val:.1f}] m).",
                )
                st.session_state["target_z"] = float(input_tz)

            with dest_col4:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button(
                    "↺ Restore Default Target",
                    width="stretch",
                    help="Restore environment's default destination coordinates",
                ):
                    st.session_state["target_x"] = float(default_goal[0])
                    st.session_state["target_y"] = float(default_goal[1])
                    st.session_state["target_z"] = float(default_goal[2])
                    st.rerun()

        selected_target = (float(input_tx), float(input_ty), float(input_tz))
        is_target_valid, target_err_msg = validate_target_coordinates(
            target=selected_target,
            bounds=active_bounds,
            start_pos=default_start,
            target_radius=target_radius_val,
            collision_radius=collision_radius_val,
        )

        if is_target_valid:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 6px; padding: 10px 14px; margin-top: 4px; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace; font-size: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="color: #10b981; font-weight: 700;">● DESTINATION ACCEPTED</span>
                        <span style="color: #64748b; margin: 0 6px;">|</span>
                        <span style="color: #94a3b8;">Target:</span> <b style="color: #00f2fe;">[{selected_target[0]:.2f}, {selected_target[1]:.2f}, {selected_target[2]:.2f}] m</b>
                        <span style="color: #64748b; margin: 0 6px;">|</span>
                        <span style="color: #94a3b8;">Arena Bounds:</span> <span style="color: #f1f5f9;">{active_bounds[0]:.1f} × {active_bounds[1]:.1f} × {active_bounds[2]:.1f} m</span>
                        <span style="color: #64748b; margin: 0 6px;">|</span>
                        <span style="color: #94a3b8;">Target Radius:</span> <span style="color: #10b981;">{target_radius_val:.2f} m</span>
                    </div>
                    <div style="color: #10b981; font-size: 11px;">
                        Simulation Ready
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.45); border-left: 5px solid #ef4444; border-radius: 6px; padding: 10px 14px; margin-top: 4px; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace; font-size: 12px;">
                    <div style="color: #ef4444; font-weight: 700; margin-bottom: 3px;">
                        🚨 INVALID DESTINATION — SIMULATION BLOCKED
                    </div>
                    <div style="color: #fca5a5; font-size: 11px;">
                        {target_err_msg}
                    </div>
                    <div style="color: #94a3b8; font-size: 10px; margin-top: 4px;">
                        Arena: {active_bounds[0]:.1f} × {active_bounds[1]:.1f} × {active_bounds[2]:.1f} m | Collision Margin: {collision_radius_val:.2f} m | Target Radius: {target_radius_val:.2f} m
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if (
            not selected_policy.startswith("Random Action Baseline")
            and is_target_valid
            and (
                abs(selected_target[0] - default_goal[0]) > 1e-4
                or abs(selected_target[1] - default_goal[1]) > 1e-4
                or abs(selected_target[2] - default_goal[2]) > 1e-4
            )
        ):
            st.markdown(
                f"<div class='baseline-disclaimer'>⚠️ <b>Generalization Notice:</b> Selected neural policy was trained "
                f"on default target ({default_goal[0]:.1f}, {default_goal[1]:.1f}, {default_goal[2]:.1f}) m. "
                f"Navigating to custom destination ({selected_target[0]:.1f}, {selected_target[1]:.1f}, {selected_target[2]:.1f}) m "
                f"may result in suboptimal trajectories or failure to reach the target.</div>",
                unsafe_allow_html=True,
            )

        # Compute run-configuration fingerprint covering all selected inputs
        # (environment, destination coords, seed, obstacle count, and policy checkpoint identity)
        policy_id = selected_policy
        if not selected_policy.startswith("Random Action Baseline") and available_models:
            raw_name = selected_policy.split(": ", 1)[-1].strip()
            matching = [p for p in available_models if p.name == raw_name]
            if matching:
                policy_id = f"{raw_name}_{matching[0].stat().st_mtime}"

        sim_fingerprint = hashlib.sha256(
            f"{env_choice}|{selected_target[0]:.2f}_{selected_target[1]:.2f}_{selected_target[2]:.2f}|{int(demo_seed)}|{int(num_obs)}|{policy_id}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        sim_key = f"flight_data_{sim_fingerprint}"
        active_fingerprint = st.session_state.get("active_sim_fingerprint")
        inputs_have_changed = (
            active_fingerprint is not None and active_fingerprint != sim_fingerprint
        )

        if inputs_have_changed:
            st.markdown(
                """
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.4); border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #fbbf24;">
                    ⚠️ <b>CONFIGURATION MODIFIED:</b> Target destination, environment model, seed, obstacles, or policy checkpoint have changed. Previous simulation results are invalidated. Click <b>'🚀 Execute Sim'</b> to simulate the active configuration.
                </div>
                """,
                unsafe_allow_html=True,
            )

        should_run_sim = launch_clicked or (active_fingerprint is None and is_target_valid)

        if should_run_sim:
            if not is_target_valid:
                st.error(f"Cannot execute simulation with invalid target: {target_err_msg}")
            else:
                with st.spinner(
                    f"Executing {env_choice} numerical dynamics simulation to [{selected_target[0]:.2f}, {selected_target[1]:.2f}, {selected_target[2]:.2f}] m..."
                ):
                    env: Any = None
                    if is_6dof:
                        env = Drone6DOFEnv(goal_pos=selected_target, num_obstacles=int(num_obs))
                    else:
                        env = DroneNavigation3DEnv(
                            goal_pos=selected_target, num_obstacles=int(num_obs)
                        )

                    try:
                        model_to_use: Optional[Any] = None

                        if (
                            not selected_policy.startswith("Random Action Baseline")
                            and available_models
                        ):
                            raw_name = selected_policy.split(": ", 1)[-1].strip()
                            matching = [p for p in available_models if p.name == raw_name]
                            if matching:
                                target_p = matching[0]
                                info = model_info_map[target_p.name]
                                if not info["valid"]:
                                    st.warning(
                                        f"Checkpoint '{raw_name}' is unsupported: {info['error']}. "
                                        "Falling back to Random Baseline."
                                    )
                                elif (
                                    info["obs_shape"] != env.observation_space.shape
                                    or info["action_shape"] != env.action_space.shape
                                ):
                                    st.warning(
                                        f"Model '{raw_name}' expects observation shape {info['obs_shape']} and "
                                        f"action shape {info['action_shape']}, but {env_choice} environment has "
                                        f"observation shape {env.observation_space.shape} and action shape "
                                        f"{env.action_space.shape}. Falling back to Random Baseline."
                                    )
                                else:
                                    if info.get("compatibility_status") == "unverified_shape_match":
                                        st.info(
                                            f"ℹ️ Model '{raw_name}' matched {env_choice} by space dimensions "
                                            f"({info['obs_shape']}/{info['action_shape']}). Training environment identity "
                                            "was inferred from shapes rather than verified metadata."
                                        )
                                    try:
                                        if info["algorithm"] == "sac":
                                            from stable_baselines3 import SAC

                                            model_to_use = SAC.load(str(target_p))
                                        else:
                                            from stable_baselines3 import PPO

                                            model_to_use = PPO.load(str(target_p))
                                    except Exception as exc:
                                        st.warning(
                                            f"Could not load checkpoint ({exc}); using random actions."
                                        )

                        run_res = run_drone_simulation_episode(
                            env=env,
                            model=model_to_use,
                            seed=int(demo_seed),
                        )
                        st.session_state["flight_data"] = run_res
                        st.session_state[sim_key] = run_res
                        st.session_state["active_sim_fingerprint"] = sim_fingerprint
                    finally:
                        env.close()

        if not inputs_have_changed and "flight_data" in st.session_state and is_target_valid:
            flight_data = st.session_state["flight_data"]
            steps = flight_data["steps"]
            total_steps = len(steps) - 1

            # Flight Outcome Status Banner
            outcome_status = flight_data["outcome"]
            is_success = outcome_status == "SUCCESS"
            is_collision = outcome_status == "COLLISION"
            is_timeout = outcome_status == "TIMEOUT"
            is_max_steps = outcome_status == "MAX_STEPS"

            if is_success:
                banner_cls = "success"
                badge_cls = "arl-pill-green"
                outcome_banner_text = "MISSION SUCCESSFUL"
            elif is_collision:
                banner_cls = "failure"
                badge_cls = "arl-pill-red"
                coll_t = flight_data.get("collision_type", "impact")
                outcome_banner_text = f"TERMINATION — COLLISION ({coll_t})"
            elif is_timeout:
                banner_cls = "timeout"
                badge_cls = "arl-pill-amber"
                outcome_banner_text = "MISSION EXPIRED — TIMEOUT"
            elif is_max_steps:
                banner_cls = "timeout"
                badge_cls = "arl-pill-amber"
                outcome_banner_text = "MISSION EXPIRED — MAX STEPS REACHED"
            else:
                banner_cls = "failure"
                badge_cls = "arl-pill-red"
                outcome_banner_text = f"TERMINATED — {outcome_status}"

            coll_notice = ""
            if flight_data.get("collision_occurred") and not is_collision:
                coll_cnt = flight_data.get("collision_count", 1)
                coll_notice = (
                    f"<span class='arl-pill arl-pill-amber'>⚠️ {coll_cnt} COLLISION(S)</span>"
                )

            final_dist = steps[-1]["distance_to_goal"]
            banner_html = (
                f'<div class="arl-outcome-banner {banner_cls}">'
                f"<div>"
                f'<div style="display: flex; align-items: center; gap: 10px;">'
                f'<span class="arl-pill {badge_cls}">● {outcome_banner_text}</span>'
                f"{coll_notice}"
                f'<span style="font-weight: 700; font-size: 15px; color: #f1f5f9;">{env_choice}</span>'
                f'<span style="color: #64748b;">|</span>'
                f'<span style="color: #94a3b8; font-size: 13px;">Policy: <b>{selected_policy.split(": ")[0]}</b></span>'
                f"</div>"
                f'<div style="margin-top: 4px; font-size: 13px; color: #cbd5e1;">'
                f"Episode finished in <b>{flight_data['total_steps']}</b> timesteps with cumulative reward <b>{flight_data['total_reward']:.2f}</b> (Seed: {flight_data['seed']})."
                f"</div>"
                f"</div>"
                f"<div style=\"text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #94a3b8;\">"
                f'Final Goal Dist: <b style="color: #f8fafc;">{final_dist:.2f} m</b><br>'
                f'Physics Mode: <b style="color: #38bdf8;">{"RK4 6-DOF" if is_6dof else "Kinematic 3-DOF"}</b>'
                f"</div>"
                f"</div>"
            )
            st.markdown(banner_html, unsafe_allow_html=True)

            # Scrubber & Viewport Control Bar
            vp_col1, vp_col2, vp_col3, vp_col4, vp_col5, vp_col6 = st.columns(
                [2.4, 0.7, 1.4, 0.7, 0.7, 0.7]
            )
            with vp_col1:
                selected_step = st.slider(
                    "Flight Step Playback Scrubber:",
                    min_value=0,
                    max_value=total_steps,
                    value=total_steps,
                    step=1,
                    help="Scrub through recorded flight timesteps to inspect 3D drone attitude, position, speed, and sensor range.",
                )
            with vp_col2:
                step_pct = (selected_step / max(1, total_steps)) * 100.0
                dt_step = 0.05
                sim_time_sec = selected_step * dt_step
                st.markdown(
                    f"""
                    <div style="padding-top: 26px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; line-height: 1.2;">
                        <b style="color: #00f2fe;">{selected_step}</b>/{total_steps} ({step_pct:.0f}%)<br>
                        <b style="color: #38bdf8;">{sim_time_sec:.2f}s</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with vp_col3:
                camera_preset = st.selectbox(
                    "Camera Perspective / Preset:",
                    options=[
                        "Arena Overview",
                        "Drone Follow",
                        "Drone Inspection",
                        "Target View",
                        "Free Orbit",
                        "Reset Camera",
                    ],
                    index=0,
                    help="Select 3D viewport camera perspective. Use 'Drone Inspection' for detailed close-up examination of quadrotor airframe.",
                )
            with vp_col4:
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                show_floor = st.checkbox(
                    "Floor", value=True, help="Render 3D flight test chamber floor and grid."
                )
            with vp_col5:
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                show_safety = st.checkbox(
                    "Safety", value=True, help="Render obstacle safety proximity zones."
                )
            with vp_col6:
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                show_traj = st.checkbox(
                    "Path", value=True, help="Render recorded 3D flight trajectory."
                )

            step_info = steps[selected_step]
            cur_pos = step_info["position"]
            cur_vel = step_info["velocity"]
            cur_lidar = step_info["lidar_ranges"]
            cur_quat = step_info.get("quaternion")

            # Telemetry calculations
            if cur_quat is not None and len(cur_quat) == 4:
                roll_rad, pitch_rad, yaw_rad = quaternion_to_euler_angles(np.asarray(cur_quat))
                roll_deg = math.degrees(roll_rad)
                pitch_deg = math.degrees(pitch_rad)
                yaw_deg = math.degrees(yaw_rad)
            else:
                roll_deg, pitch_deg, yaw_deg = 0.0, 0.0, 0.0

            omega_mag = step_info.get("angular_speed", 0.0)
            ang_vel = step_info.get("angular_velocity")
            if ang_vel is not None and len(ang_vel) == 3:
                p_val, q_val, r_val = ang_vel
            else:
                p_val, q_val, r_val = 0.0, 0.0, 0.0

            # Two-Column Master View: 3D Viewport on Left (dominant), Telemetry Console on Right
            col_viewport, col_telemetry = st.columns([2.5, 1.0])

            with col_viewport:
                # Build Interactive 3D Plotly Arena
                fig = build_arena_3d_figure(
                    bounds=flight_data["bounds"],
                    obstacles=flight_data["obstacles"],
                    target=flight_data["target"],
                    start_pos=flight_data["start_pos"],
                    trajectory=[s["position"] for s in steps[: selected_step + 1]],
                    current_pos=cur_pos,
                    current_vel=cur_vel,
                    attitude_quaternion=cur_quat,
                    lidar_rays=flight_data["lidar_rays"],
                    lidar_ranges=cur_lidar,
                    show_lidar=show_lidar_rays,
                    title=f"3D Flight Arena ({env_choice}) — Step {selected_step}/{total_steps} (Distance to Goal: {step_info['distance_to_goal']:.2f}m)",
                    camera_preset=camera_preset,
                    show_floor=show_floor,
                    show_safety_zones=show_safety,
                    show_trajectory=show_traj,
                    propeller_phase=selected_step * 0.45,
                    arm_span=flight_data.get("arm_span", 0.4),
                    arm_length=flight_data.get("arm_length", 0.2),
                    rotor_radius=flight_data.get("rotor_radius", 0.09),
                    target_radius=flight_data.get("target_radius", 1.5),
                    body_frame_lidar=flight_data.get("body_frame_lidar", False),
                )
                # Give viewport substantial height for dominance
                fig.update_layout(height=660)
                st.plotly_chart(fig, width="stretch")

            with col_telemetry:
                # 1. NAVIGATION TELEMETRY
                rel_target = flight_data["target"] - cur_pos
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-header">📍 Navigation & Position</div>
                        <div class="telemetry-grid">
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Altitude (Z)</span>
                                <span class="telemetry-stat-val cyan">{cur_pos[2]:.2f}<span class="unit">m</span></span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Range to Goal</span>
                                <span class="telemetry-stat-val green">{step_info["distance_to_goal"]:.2f}<span class="unit">m</span></span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Nearest Obstacle</span>
                                <span class="telemetry-stat-val {"red" if step_info["min_obstacle_distance"] < 1.0 else "amber"}">{step_info["min_obstacle_distance"]:.2f}<span class="unit">m</span></span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Rel Target [X,Y]</span>
                                <span class="telemetry-stat-val">[{rel_target[0]:.1f}, {rel_target[1]:.1f}]</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 2. FLIGHT DYNAMICS
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-header">⚡ Flight Dynamics</div>
                        <div class="telemetry-grid">
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Linear Speed</span>
                                <span class="telemetry-stat-val cyan">{step_info["speed"]:.2f}<span class="unit">m/s</span></span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Angular Speed</span>
                                <span class="telemetry-stat-val">{omega_mag:.2f}<span class="unit">rad/s</span></span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Velocity [Vx, Vy]</span>
                                <span class="telemetry-stat-val">[{cur_vel[0]:.1f}, {cur_vel[1]:.1f}]</span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Climb Rate (Vz)</span>
                                <span class="telemetry-stat-val">{cur_vel[2]:+.2f}<span class="unit">m/s</span></span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 3. ATTITUDE & ORIENTATION (6-DOF) / KINEMATICS (3-DOF)
                if is_6dof and cur_quat is not None:
                    st.markdown(
                        f"""
                        <div class="telemetry-card">
                            <div class="telemetry-header">🧭 Attitude & Orientation (6-DOF)</div>
                            <div class="telemetry-grid">
                                <div class="telemetry-stat">
                                    <span class="telemetry-stat-label">Roll (φ)</span>
                                    <span class="telemetry-stat-val">{roll_deg:+.1f}<span class="unit">°</span></span>
                                </div>
                                <div class="telemetry-stat">
                                    <span class="telemetry-stat-label">Pitch (θ)</span>
                                    <span class="telemetry-stat-val">{pitch_deg:+.1f}<span class="unit">°</span></span>
                                </div>
                                <div class="telemetry-stat">
                                    <span class="telemetry-stat-label">Yaw (ψ)</span>
                                    <span class="telemetry-stat-val">{yaw_deg:+.1f}<span class="unit">°</span></span>
                                </div>
                                <div class="telemetry-stat">
                                    <span class="telemetry-stat-label">Body Rates [p,q,r]</span>
                                    <span class="telemetry-stat-val">[{p_val:.1f}, {q_val:.1f}, {r_val:.1f}]</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="telemetry-card">
                            <div class="telemetry-header">🧭 Kinematic State (3-DOF)</div>
                            <div style="font-size: 11px; color: #94a3b8; line-height: 1.4;">
                                <b>Point-Mass Kinematics:</b> Continuous 3D acceleration control. Rotational torque and attitude dynamics are not simulated in 3-DOF mode.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 4. SENSORS & LIDAR
                min_lidar = float(np.min(cur_lidar)) if len(cur_lidar) > 0 else 0.0
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-header">📡 Sensors & Perception</div>
                        <div class="telemetry-grid">
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">LiDAR Rays</span>
                                <span class="telemetry-stat-val cyan">16 <span class="unit">Beams</span></span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Closest Return</span>
                                <span class="telemetry-stat-val {"red" if min_lidar < 1.5 else "green"}">{min_lidar:.2f}<span class="unit">m</span></span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 5. REWARDS & PERFORMANCE
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-header">🎯 Performance & Rewards</div>
                        <div class="telemetry-grid">
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Step Reward</span>
                                <span class="telemetry-stat-val">{step_info["reward"]:+.2f}</span>
                            </div>
                            <div class="telemetry-stat">
                                <span class="telemetry-stat-label">Cumulative Return</span>
                                <span class="telemetry-stat-val green">{step_info["cumulative_reward"]:+.2f}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Expandable Simulation Diagnostics Section
            with st.expander("🔍 Simulation Diagnostics & Physics Specifications"):
                diag_col1, diag_col2 = st.columns(2)
                with diag_col1:
                    st.markdown(
                        """
                        **Dynamics Engine Specifications**
                        - **6-DOF Quadrotor Dynamics**: Runge-Kutta 4th Order (RK4) integration with 5 substeps ($\\Delta t_{\\text{sub}} = 0.01\\text{ s}$).
                        - **State Representation**: 36-dimensional state vector with position, linear velocity, unit quaternion attitude $\\mathbf{q} = [q_w, q_x, q_y, q_z]$, and body angular velocity $\\boldsymbol{\\omega} = [p, q, r]$.
                        - **Control**: Continuous 4-motor thrust commands mapped to physical body torques and collective thrust.
                        """
                    )
                with diag_col2:
                    st.markdown(
                        """
                        **Sensor & Obstacle Parameters**
                        - **Spherical LiDAR**: 16 analytical raycast rangefinders with geometric ray-sphere and bounding box intersection calculations.
                        - **Collision Detection**: Continuous sphere-to-sphere collision checking with obstacle clearance radii.
                        - **Playback Notice**: Recorded trajectory playback from deterministic numerical evaluation episodes.
                        """
                    )
        else:
            st.info(
                "💡 Please enter valid target coordinates within the arena boundaries and click **🚀 Execute Sim** to launch numerical simulation."
            )

    # ==========================================
    # TAB 2: TRAIN POLICY (PPO & SAC)
    # ==========================================
    with tabs[1]:
        st.subheader("Train Autonomous Flight Policy (PPO & SAC)")
        st.markdown(
            "Execute genuine reinforcement learning training on your CPU with real-time "
            "progress streaming, step throughput, and episodic reward tracking."
        )

        tr_col1, tr_col2, tr_col3, tr_col4 = st.columns(4)
        with tr_col1:
            train_algo = st.selectbox(
                "Algorithm:",
                options=["PPO", "SAC"],
                index=0,
                help="PPO: On-policy clipped surrogate. SAC: Off-policy maximum-entropy actor-critic.",
            )
        with tr_col2:
            train_env_choice = st.selectbox(
                "Dynamics Model:",
                options=["3-DOF Kinematic Drone", "6-DOF Quadrotor"],
                index=0,
                help="Select target physics simulation environment.",
            )
        with tr_col3:
            train_steps = st.select_slider(
                "Training Budget (Timesteps):",
                options=[2000, 5000, 10000, 25000, 50000],
                value=5000,
                help="Timestep budget. 5,000-10,000 steps completes rapidly for CPU interactive testing.",
            )
        with tr_col4:
            train_seed = st.number_input("Random Seed:", value=42, step=1)

        # Training Target Destination Configuration
        st.markdown("**Target Destination for Training:**")
        tr_tcol1, tr_tcol2, tr_tcol3, tr_tcol4 = st.columns([1.1, 1.1, 1.1, 1.5])
        with tr_tcol1:
            train_tx = st.number_input(
                "Train Target X (m):",
                value=25.0,
                min_value=-15.0,
                max_value=45.0,
                step=1.0,
                format="%.2f",
                key="train_tx",
            )
        with tr_tcol2:
            train_ty = st.number_input(
                "Train Target Y (m):",
                value=25.0,
                min_value=-15.0,
                max_value=45.0,
                step=1.0,
                format="%.2f",
                key="train_ty",
            )
        with tr_tcol3:
            train_tz = st.number_input(
                "Train Target Z (m):",
                value=10.0,
                min_value=-15.0,
                max_value=30.0,
                step=1.0,
                format="%.2f",
                key="train_tz",
            )
        with tr_tcol4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            use_custom_train_target = st.checkbox(
                "Train with Custom Target",
                value=False,
                key="use_custom_train_target",
                help="Train against fixed custom target instead of default environment target.",
            )

        train_target = (
            (float(train_tx), float(train_ty), float(train_tz))
            if use_custom_train_target
            else (25.0, 25.0, 10.0)
        )
        is_train_target_valid, train_target_err = validate_target_coordinates(
            target=train_target,
            bounds=(30.0, 30.0, 15.0),
            start_pos=(5.0, 5.0, 5.0),
        )
        if use_custom_train_target:
            if is_train_target_valid:
                st.info(
                    f"🎯 **Fixed Custom Target Training Mode:** All training episodes will use destination [{train_target[0]:.1f}, {train_target[1]:.1f}, {train_target[2]:.1f}] m. "
                    "Note: The policy will specialize for this specific waypoint and is not guaranteed to generalize to other destinations."
                )
            else:
                st.error(f"Invalid Training Destination: {train_target_err}")

        # Hyperparameter configuration
        with st.expander("⚙️ Algorithm Hyperparameters", expanded=False):
            hp_c1, hp_c2, hp_c3 = st.columns(3)
            with hp_c1:
                train_lr = st.selectbox(
                    "Learning Rate:",
                    options=[0.0001, 0.0003, 0.001],
                    index=1,
                )
                train_gamma = st.number_input(
                    "Discount Factor (γ):", value=0.99, min_value=0.8, max_value=1.0, step=0.01
                )
            with hp_c2:
                if train_algo == "PPO":
                    train_n_steps = st.number_input(
                        "Rollout Steps (n_steps):",
                        value=1024,
                        min_value=64,
                        max_value=4096,
                        step=128,
                    )
                    train_batch_size = st.number_input(
                        "Minibatch Size:", value=64, min_value=16, max_value=512, step=16
                    )
                else:
                    train_buffer_size = st.number_input(
                        "Replay Buffer Size:",
                        value=50000,
                        min_value=1000,
                        max_value=200000,
                        step=5000,
                    )
                    train_batch_size = st.number_input(
                        "Minibatch Size:", value=128, min_value=16, max_value=512, step=32
                    )
            with hp_c3:
                if train_algo == "PPO":
                    train_n_epochs = st.number_input(
                        "Epochs (n_epochs):", value=10, min_value=1, max_value=30, step=1
                    )
                    train_clip_range = st.number_input(
                        "Clip Range:", value=0.2, min_value=0.05, max_value=0.4, step=0.05
                    )
                    train_ent_coef = st.number_input(
                        "Entropy Coef:", value=0.0, min_value=0.0, max_value=0.1, step=0.01
                    )
                else:
                    train_learning_starts = st.number_input(
                        "Learning Starts:", value=100, min_value=0, max_value=5000, step=50
                    )
                    train_tau = st.number_input(
                        "Target Smoothing (tau):",
                        value=0.005,
                        min_value=0.001,
                        max_value=0.1,
                        step=0.005,
                        format="%.3f",
                    )
                    train_ent_coef_str = st.selectbox(
                        "Entropy Regularization:", ["auto", "0.01", "0.05", "0.1"]
                    )

        is_training = st.session_state.get("is_training", False)
        start_train_btn = st.button(
            f"▶ Start {train_algo} Training",
            type="primary",
            disabled=(is_training or (use_custom_train_target and not is_train_target_valid)),
        )

        if start_train_btn:
            st.session_state["is_training"] = True
            st.info(
                f"Training {train_algo} on {train_env_choice} for {train_steps:,} timesteps on CPU..."
            )
            prog_bar = st.progress(0.0)
            train_status_box: Any = st.empty()

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_steps = m_c1.empty()
            m_eps = m_c2.empty()
            m_rew = m_c3.empty()
            m_speed = m_c4.empty()

            try:
                env_id = "drone-6dof" if train_env_choice == "6-DOF Quadrotor" else "drone"
                algo_key = train_algo.lower()
                experiment_name = f"{env_id.replace('-', '_')}_{algo_key}_gui"

                algo_params: Dict[str, Any] = {}
                if algo_key == "ppo":
                    algo_params = {
                        "n_steps": int(train_n_steps),
                        "n_epochs": int(train_n_epochs),
                        "clip_range": float(train_clip_range),
                        "ent_coef": float(train_ent_coef),
                    }
                else:
                    sac_ent: Union[str, float] = (
                        "auto" if train_ent_coef_str == "auto" else float(train_ent_coef_str)
                    )
                    algo_params = {
                        "buffer_size": int(train_buffer_size),
                        "learning_starts": int(train_learning_starts),
                        "tau": float(train_tau),
                        "ent_coef": sac_ent,
                    }

                env_params: Dict[str, Any] = {}
                if use_custom_train_target and is_train_target_valid:
                    env_params["goal_pos"] = [
                        float(train_target[0]),
                        float(train_target[1]),
                        float(train_target[2]),
                    ]

                train_cfg = ExperimentConfig(
                    name=experiment_name,
                    seed=int(train_seed),
                    algorithm=AlgorithmConfig(
                        name=algo_key,
                        learning_rate=float(train_lr),
                        gamma=float(train_gamma),
                        batch_size=int(train_batch_size),
                        parameters=algo_params,
                    ),
                    environment=EnvironmentConfig(
                        name=env_id,
                        parameters=env_params,
                    ),
                    training=TrainingConfig(
                        total_timesteps=int(train_steps),
                        checkpoint_freq=0,
                    ),
                    output_dir=Path("artifacts"),
                )

                def _on_gui_progress(data: Dict[str, Any]) -> None:
                    prog_bar.progress(data["progress_ratio"])
                    rew_disp = (
                        f"{data['rolling_reward']:.2f}"
                        if data["rolling_reward"] is not None
                        else "--"
                    )
                    train_status_box.text(
                        f"Status: {data['status'].upper()} | Step {data['current_step']:,} / "
                        f"{data['total_timesteps']:,} ({data['progress_ratio'] * 100:.1f}%) | "
                        f"Speed: {data['fps']:.0f} steps/s"
                    )
                    m_steps.metric("Timesteps", f"{data['current_step']:,}")
                    m_eps.metric("Episodes", f"{data['episodes']}")
                    m_rew.metric("Rolling Return", rew_disp)
                    m_speed.metric("Throughput", f"{data['fps']:.0f} steps/s")

                gui_callback = TrainingProgressCallback(
                    total_timesteps=int(train_steps),
                    on_update=_on_gui_progress,
                    update_frequency=100,
                )

                trainer = RLTrainer(config=train_cfg, callbacks=[gui_callback])
                t0 = time.time()
                result = trainer.fit()
                duration = time.time() - t0

                prog_bar.progress(1.0)
                train_status_box.text(f"Training Complete! Total duration: {duration:.2f}s")
                st.success(f"Checkpoint saved: `{result.final_model_path}`")

                st.session_state["latest_training_rewards"] = result.episode_rewards

                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Total Steps", f"{result.total_timesteps:,}")
                t2.metric("Episodes Completed", f"{result.episodes_completed}")
                t3.metric("Training Time", f"{duration:.2f}s")
                t4.metric("Mean Final Reward", f"{result.mean_reward:.2f}")

            except Exception as e:
                st.error(f"Training failed: {e}")
            finally:
                st.session_state["is_training"] = False

        # Display Training Curve
        rewards_to_show = st.session_state.get("latest_training_rewards", [])
        if not rewards_to_show:
            demo_meta_path = Path("artifacts/metadata/drone_ppo_demo_training.json")
            if demo_meta_path.exists():
                import json

                try:
                    with open(demo_meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        rewards_to_show = data.get("episode_rewards", [])
                except Exception:
                    pass

        if rewards_to_show:
            st.markdown("### Training Progress Curve")
            fig_curve = build_training_curve_figure(
                episode_rewards=rewards_to_show,
                window_size=10,
                title="PPO Episode Return over Training Episodes",
            )
            st.plotly_chart(fig_curve, width="stretch")

    # ==========================================
    # TAB 3: EVALUATION & BASELINE
    # ==========================================
    with tabs[2]:
        st.subheader("Policy Evaluation & Random Baseline Comparison")
        st.markdown(
            "Empirically compare trained **RL policies (PPO & SAC)** against an **untrained Random Policy baseline** "
            "under identical environment seeds, obstacles, destination targets, and episode counts."
        )

        e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns([1.8, 0.8, 0.8, 0.8, 1.2])
        with e_col1:
            eval_model_name = st.selectbox(
                "Evaluation Model:",
                options=model_names if model_names else ["No models found in artifacts/models"],
                index=0,
            )
        with e_col2:
            eval_episodes = st.slider("Episodes:", min_value=5, max_value=50, value=20, step=5)
        with e_col3:
            eval_base_seed = st.number_input(
                "Base Seed:", min_value=0, max_value=999999, value=42, step=1
            )
        with e_col4:
            eval_num_obs = st.slider("Obstacles:", min_value=0, max_value=8, value=4, step=1)
        with e_col5:
            eval_mode = st.radio(
                "Mode:",
                ["Compare Both (Trained vs Random)", "Trained Policy Only", "Random Policy Only"],
            )

        # Evaluation Destination Configuration
        st.markdown("**Target Destination for Evaluation:**")
        ev_tcol1, ev_tcol2, ev_tcol3, ev_tcol4 = st.columns([1.1, 1.1, 1.1, 1.5])
        with ev_tcol1:
            eval_tx = st.number_input(
                "Eval Target X (m):",
                value=25.0,
                min_value=-15.0,
                max_value=45.0,
                step=1.0,
                format="%.2f",
                key="eval_tx",
            )
        with ev_tcol2:
            eval_ty = st.number_input(
                "Eval Target Y (m):",
                value=25.0,
                min_value=-15.0,
                max_value=45.0,
                step=1.0,
                format="%.2f",
                key="eval_ty",
            )
        with ev_tcol3:
            eval_tz = st.number_input(
                "Eval Target Z (m):",
                value=10.0,
                min_value=-15.0,
                max_value=30.0,
                step=1.0,
                format="%.2f",
                key="eval_tz",
            )
        with ev_tcol4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            use_custom_eval_target = st.checkbox(
                "Evaluate with Custom Target",
                value=False,
                key="use_custom_eval_target",
                help="Evaluate both trained policy and baseline on custom target coordinates.",
            )

        eval_target = (
            (float(eval_tx), float(eval_ty), float(eval_tz))
            if use_custom_eval_target
            else (25.0, 25.0, 10.0)
        )
        is_eval_target_valid, eval_target_err = validate_target_coordinates(
            target=eval_target,
            bounds=(30.0, 30.0, 15.0),
            start_pos=(5.0, 5.0, 5.0),
        )
        if use_custom_eval_target:
            if is_eval_target_valid:
                st.info(
                    f"🎯 **Identical Custom Target Evaluation Active:** Both policy and random baseline will navigate toward [{eval_target[0]:.1f}, {eval_target[1]:.1f}, {eval_target[2]:.1f}] m."
                )
            else:
                st.error(f"Invalid Evaluation Destination: {eval_target_err}")

        run_eval_btn = st.button("📊 Run Benchmark Evaluation", type="primary")

        if run_eval_btn:
            if use_custom_eval_target and not is_eval_target_valid:
                st.error(f"Cannot run evaluation with invalid target: {eval_target_err}")
            else:
                with st.spinner(f"Running evaluation ({eval_episodes} episodes)..."):
                    is_eval_6dof = False
                    target_path = None
                    info = None
                    if available_models and eval_model_name in model_info_map:
                        target_path = next(
                            (p for p in available_models if p.name == eval_model_name), None
                        )
                        if target_path:
                            info = model_info_map[target_path.name]
                            if info.get("compatible_env") == "6-DOF":
                                is_eval_6dof = True

                    eval_env = (
                        Drone6DOFEnv(
                            max_steps=200,
                            num_obstacles=int(eval_num_obs),
                            goal_pos=eval_target if use_custom_eval_target else None,
                        )
                        if is_eval_6dof
                        else DroneNavigation3DEnv(
                            max_steps=200,
                            num_obstacles=int(eval_num_obs),
                            goal_pos=eval_target if use_custom_eval_target else None,
                        )
                    )
                try:
                    metrics_to_plot: Dict[str, Any] = {}

                    # Evaluate Model
                    if eval_mode in ["Compare Both (Trained vs Random)", "Trained Policy Only"]:
                        if target_path is not None and info is not None:
                            if not info["valid"]:
                                st.warning(
                                    f"Checkpoint '{eval_model_name}' is unsupported: {info['error']}"
                                )
                            elif (
                                info["obs_shape"] != eval_env.observation_space.shape
                                or info["action_shape"] != eval_env.action_space.shape
                            ):
                                st.warning(
                                    f"Checkpoint '{eval_model_name}' expects observation shape {info['obs_shape']} "
                                    f"and action shape {info['action_shape']}, but active evaluation environment requires "
                                    f"{eval_env.observation_space.shape} and {eval_env.action_space.shape}. "
                                    "Cannot evaluate incompatible model."
                                )
                            else:
                                trained_algo = load_algorithm_from_pretrained(
                                    target_path, env=eval_env
                                )

                                trained_res = evaluate_policy(
                                    algorithm=trained_algo,
                                    env=eval_env,
                                    num_episodes=int(eval_episodes),
                                    base_seed=int(eval_base_seed),
                                )
                                algo_title = (
                                    f"{info['algorithm'].upper()} Policy ({'6-DOF' if is_eval_6dof else '3-DOF'})"
                                    if info["algorithm"]
                                    else "Trained Policy"
                                )
                                metrics_to_plot[algo_title] = trained_res
                                Evaluator.save_report(
                                    trained_res,
                                    Path("artifacts/evaluation.json"),
                                    target_pos=eval_target if use_custom_eval_target else None,
                                )
                        else:
                            st.warning(
                                "No saved model found. Please train a compatible agent first."
                            )

                    # Evaluate Random Policy
                    if eval_mode in ["Compare Both (Trained vs Random)", "Random Policy Only"]:
                        rand_res = evaluate_random_policy(
                            env=eval_env,
                            num_episodes=int(eval_episodes),
                            base_seed=int(eval_base_seed),
                        )
                        metrics_to_plot[
                            f"Random Policy ({'6-DOF' if is_eval_6dof else '3-DOF'})"
                        ] = rand_res

                    st.session_state["benchmark_results"] = metrics_to_plot

                except Exception as exc:
                    st.error(f"Evaluation error: {exc}")
                finally:
                    eval_env.close()

        # Display Benchmark Results
        bench_results = st.session_state.get("benchmark_results", {})
        if bench_results:
            st.markdown("### Empirical Benchmark Results")

            # Metrics Table
            table_data = []
            for pol_name, m in bench_results.items():
                succ_str = f"{m.success_rate * 100:.1f}%" if m.success_rate is not None else "N/A"
                coll_str = (
                    f"{m.collision_rate * 100:.1f}%" if m.collision_rate is not None else "N/A"
                )
                table_data.append(
                    {
                        "Policy": pol_name,
                        "Episodes": m.episodes,
                        "Success Rate": succ_str,
                        "Collision Rate": coll_str,
                        "Mean Reward": f"{m.mean_reward:.2f} ± {m.std_reward:.2f}",
                        "Mean Episode Length": f"{m.mean_episode_length:.1f} steps",
                    }
                )
            st.table(table_data)

            # Side-by-Side Comparison Chart
            fig_comp = build_comparison_bar_chart(
                metrics_map=bench_results,
                title="Trained Policy vs Random Baseline (Success & Collision Rates)",
            )
            st.plotly_chart(fig_comp, width="stretch")
            st.info("Evaluation report exported to `artifacts/evaluation.json`")
        else:
            st.caption("Click 'Run Benchmark Evaluation' to generate empirical metrics.")

    # ==========================================
    # TAB 4: DIFFICULTY EXPERIMENT
    # ==========================================
    with tabs[3]:
        st.subheader("Obstacle-Density Experiment")
        st.markdown(
            "This experiment demonstrates how environment difficulty impacts agent performance. "
            "The policy is evaluated under identical seeds across **4, 6, and 8 procedural obstacles**."
        )
        st.info(
            "💡 **Core Insight**: *More obstacles increase the navigation difficulty of the environment.*"
        )

        d_col1, d_col2 = st.columns([2, 1])
        with d_col1:
            diff_model_name = st.selectbox(
                "Model to Test:",
                options=model_names if model_names else ["No models found in artifacts/models"],
                index=0,
                key="diff_model",
            )
        with d_col2:
            diff_episodes = st.slider(
                "Episodes per Condition:", min_value=5, max_value=20, value=10, step=5
            )

        run_diff_btn = st.button("🎯 Run Obstacle-Density Experiment", type="primary")

        if run_diff_btn:
            if available_models and diff_model_name in model_info_map:
                target_path = next(p for p in available_models if p.name == diff_model_name)
                info = model_info_map[target_path.name]
                if not info["valid"]:
                    st.warning(f"Checkpoint '{diff_model_name}' is unsupported: {info['error']}")
                elif info["obs_shape"] != (29,) or info["action_shape"] != (3,):
                    st.warning(
                        f"Checkpoint '{diff_model_name}' expects observation shape {info['obs_shape']} and "
                        f"action shape {info['action_shape']}, which is incompatible with the 3-DOF obstacle-density "
                        "experiment (requires observation shape (29,) and action shape (3,))."
                    )
                else:
                    with st.spinner("Evaluating across 4, 6, and 8 obstacles..."):
                        dummy_env = DroneNavigation3DEnv()
                        try:
                            algo = load_algorithm_from_pretrained(target_path, env=dummy_env)
                            density_results = run_obstacle_density_experiment(
                                algorithm=algo,
                                obstacle_counts=(4, 6, 8),
                                episodes_per_density=int(diff_episodes),
                                base_seed=42,
                                output_path=Path("artifacts/obstacle_density_experiment.json"),
                            )
                            st.session_state["density_results"] = density_results
                        finally:
                            dummy_env.close()
            else:
                st.warning("No saved model found. Please train a compatible agent first.")

        density_data = st.session_state.get("density_results", None)
        if density_data:
            st.markdown("### Measured Experiment Data")
            display_rows = []
            for row in density_data:
                display_rows.append(
                    {
                        "Obstacle Count": f"{row['obstacle_count']} Obstacles",
                        "Episodes": row["episodes"],
                        "Success Rate": f"{row['success_rate'] * 100:.1f}%",
                        "Collision Rate": f"{row['collision_rate'] * 100:.1f}%",
                        "Mean Reward": f"{row['mean_reward']:.2f}",
                        "Mean Steps": f"{row['mean_episode_length']:.1f}",
                    }
                )
            st.table(display_rows)

            fig_density = build_density_experiment_figure(
                density_results=density_data,
                title="Performance Degradation under Increasing Obstacle Density",
            )
            st.plotly_chart(fig_density, width="stretch")
            st.success(
                "Experiment results exported to `artifacts/obstacle_density_experiment.json`"
            )
        else:
            st.warning(
                "Not evaluated — click 'Run Obstacle-Density Experiment' to collect actual data."
            )

    # ==========================================
    # TAB 5: ABOUT & ARCHITECTURE
    # ==========================================
    with tabs[4]:
        st.subheader("System Architecture & Specifications")

        st.markdown(
            r"""
            ### Simulation Environments: 3-DOF vs. 6-DOF Comparison

            | Property | 3-DOF | 6-DOF |
            |---|---|---|
            | Observation dimensions | 29 | 36 |
            | Action dimensions | 3 | 4 |
            | Control | Acceleration | Motor thrust |
            | Attitude | Not modeled | Quaternion |
            | Angular velocity | Not modeled | 3-axis |

            ### How It Works (Step-by-Step)
            1. **Observation**: State vector received from 3-DOF point-mass (29 dims) or 6-DOF rigid-body (36 dims).
            2. **Policy Prediction**: Trained neural network (PPO or SAC) predicts continuous control actions.
            3. **Action Execution**: Commanded accelerations (3-DOF) or individual motor thrusts (6-DOF) applied to dynamics.
            4. **Physics Integration**: Semi-implicit Euler (3-DOF) or Runge-Kutta 4th-order (RK4) integration with quaternion kinematics (6-DOF).
            5. **Reward Feedback**: Progress towards goal rewarded; collisions, control effort, and instability penalized.
            6. **Policy Learning**: Agents trained with Stable-Baselines3 PPO or SAC on parameterized Gymnasium environments.
            """
        )

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown(
                r"""
                ### 3-DOF Kinematic Drone Specs
                - **Observation Space (29 Dims)**:
                  - Drone position $\mathbf{p} = [x, y, z]$ (3 dims)
                  - Velocity vector $\mathbf{v} = [v_x, v_y, v_z]$ (3 dims)
                  - Target position $\mathbf{g} = [g_x, g_y, g_z]$ (3 dims)
                  - Relative target offset $\mathbf{g} - \mathbf{p}$ (3 dims)
                  - Target distance ratio (1 dim)
                  - 16-ray spherical LiDAR rangefinder readings (16 dims)
                - **Action Space (3 Dims)**:
                  - Continuous 3D acceleration commands in $[-1.0, 1.0]^3$
                - **Integration**: Semi-implicit Euler ($\Delta t = 0.05\text{ s}$)
                """
            )

        with arch_col2:
            st.markdown(
                r"""
                ### 6-DOF Rigid-Body Quadrotor Specs
                - **Observation Space (36 Dims)**:
                  - Position (3 dims), Velocity (3 dims)
                  - Attitude quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]$ (4 dims)
                  - Body angular velocity $\boldsymbol{\omega} = [p, q, r]$ (3 dims)
                  - Target position (3 dims), Relative offset (3 dims), Distance ratio (1 dim)
                  - 16-ray spherical LiDAR rangefinder readings (16 dims)
                - **Action Space (4 Dims)**:
                  - 4 continuous rotor thrust commands in $[-1.0, 1.0]^4 \to [0, F_{\max}]^4$
                - **Integration**: RK4 with 5 substeps ($\Delta t_{\text{sub}} = 0.01\text{ s}$) and quaternion re-normalization
                """
            )

        st.markdown(
            r"""
            ### User-Defined 3D Destination & Validation System
            - **Arbitrary 3D Coordinates**: Users can define target destinations $\mathbf{g} = [g_x, g_y, g_z]$ in metres for both 3-DOF kinematic and 6-DOF dynamic simulations.
            - **Strict Pre-Flight Validation**:
              - **Finite Coordinates**: Requires three finite numbers (rejects NaN, $+\infty$, $-\infty$).
              - **Arena Boundaries**: Validates coordinates within $[0, X_{\max}] \times [0, Y_{\max}] \times [0, Z_{\max}]$ ($30 \times 30 \times 15\text{ m}$).
              - **Boundary Collision Margins**: Requires clearance from perimeter walls exceeding aircraft collision radius ($r_{\text{collision}} = 0.8\text{ m}$).
              - **Launch Clearance**: Rejects destinations within arrival radius ($r_{\text{target}} = 1.5\text{ m}$) of launch position ($[5, 5, 5]\text{ m}$).
            - **Closed-Loop Physics Integration**: Selected coordinates are instantiated directly in environment state ($\mathbf{g}$), updating relative observation vectors, distance metrics, obstacle generation clearance, and arrival reward triggers.
            - **Training & Generalization Advisory**: RL policies trained on fixed destinations specialize on that waypoint vector. Testing policies on custom destinations tests out-of-distribution transfer; success is not guaranteed without generalized multi-target training.
            """
        )

        st.markdown("---")
        st.markdown(
            """
            ### Educational Scope & Honest Technical Limitations
            - **Educational Platform**: Designed for algorithmic benchmarking and educational demonstration.
            - **Theoretical Simulation**: Canonical rigid-body quadrotor and point-mass physics without hardware validation claims.
            - **Analytical Raycasts**: The 16-ray spherical LiDAR calculates geometric ray-sphere and ray-box intersections.
            - **No Real Drone Autopilot**: Not intended for direct deployment on physical UAV flight controllers (PX4/ArduPilot).
            """
        )

    # ==========================================
    # TAB 6: FLIGHT SCENARIOS & BENCHMARKS
    # ==========================================
    with tabs[5]:
        st.subheader("Autonomous Flight Validation & Empirical Benchmarks")
        st.markdown(
            "Execute reproducible **10-scenario flight validation tests** and standardized "
            "**PPO / SAC comparative benchmarks** across 3-DOF and 6-DOF quadrotor dynamics models."
        )

        sub_view = st.radio(
            "Select Validation Module:",
            [
                "🎯 Autonomous Flight Scenarios (Standard & Advanced)",
                "📊 PPO & SAC Benchmark Suite",
            ],
            horizontal=True,
        )

        if sub_view == "🎯 Autonomous Flight Scenarios (Standard & Advanced)":
            from adaptive_rl.controllers import QuadrotorLQRController
            from adaptive_rl.flight_testing.scenarios import get_all_flight_scenarios

            scenarios = get_all_flight_scenarios()
            scenario_names = [sc.name for sc in scenarios]

            sc_col1, sc_col2 = st.columns([1.5, 1.0])
            with sc_col1:
                selected_sc_name = st.selectbox(
                    "Flight Scenario:",
                    options=scenario_names,
                    index=0,
                    help="Select one of the 18 standardized and advanced 3D autonomous flight scenarios.",
                )
            selected_sc = next(sc for sc in scenarios if sc.name == selected_sc_name)

            with sc_col2:
                ctrl_options = [
                    "Reference Flight Controller (Cascaded PID)",
                    "Optimal Flight Controller (LQR)",
                ] + [f"Model: {name}" for name in model_names]
                selected_ctrl_name = st.selectbox(
                    "Flight Controller / Policy:",
                    options=ctrl_options,
                    index=0,
                )

            st.info(f"**Scenario Objective:** {selected_sc.description}")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric(
                "Launch Position",
                f"[{selected_sc.initial_position[0]:.1f}, {selected_sc.initial_position[1]:.1f}, {selected_sc.initial_position[2]:.1f}] m",
            )
            final_wp = (
                selected_sc.waypoints[-1] if selected_sc.waypoints else selected_sc.initial_position
            )
            m_col2.metric(
                "Target Waypoint", f"[{final_wp[0]:.1f}, {final_wp[1]:.1f}, {final_wp[2]:.1f}] m"
            )
            total_obs = len(selected_sc.obstacles) + len(selected_sc.dynamic_obstacles)
            m_col3.metric("Obstacles", f"{total_obs} spheres")
            m_col4.metric(
                "Max Steps", f"{selected_sc.max_steps} ({selected_sc.max_steps * 0.05:.1f}s)"
            )

            if st.button("🚀 Execute Autonomous Flight Scenario", type="primary"):
                with st.spinner(
                    "Simulating closed-loop 6-DOF flight dynamics with Runge-Kutta 4..."
                ):
                    # Resolve controller
                    controller_inst: Any = None
                    if selected_ctrl_name == "Reference Flight Controller (Cascaded PID)":
                        controller_inst = QuadrotorPIDController(dt=0.05)
                    elif selected_ctrl_name == "Optimal Flight Controller (LQR)":
                        controller_inst = QuadrotorLQRController(dt=0.05)
                    else:
                        model_key = selected_ctrl_name.replace("Model: ", "")
                        m_path = next((m for m in available_models if m.name == model_key), None)
                        if m_path:
                            try:
                                chk_info = inspect_model_checkpoint(m_path)
                                if chk_info.get("algorithm") == "sac":
                                    from stable_baselines3 import SAC

                                    controller_inst = SAC.load(str(m_path))
                                else:
                                    from stable_baselines3 import PPO

                                    controller_inst = PPO.load(str(m_path))
                            except Exception as exc:
                                st.warning(
                                    f"Failed to load checkpoint ({exc}). Falling back to Reference PID Controller."
                                )
                                controller_inst = QuadrotorPIDController(dt=0.05)

                    runner = FlightScenarioRunner()
                    res = runner.run_scenario(selected_sc, controller=controller_inst)
                    st.session_state["scenario_result"] = res

            if "scenario_result" in st.session_state:
                res = st.session_state["scenario_result"]
                metrics = res.metrics

                # Outcome banner
                outcome_cls = "arl-pill-green" if metrics.success else "arl-pill-red"
                st.markdown(
                    f"""
                    <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 18px; margin: 12px 0;">
                        <span class="arl-pill {outcome_cls}">● OUTCOME: {metrics.outcome.upper()}</span>
                        <span style="color: #cbd5e1; margin-left: 12px; font-size: 14px;">
                            Completed in <b>{metrics.duration_steps}</b> steps (<b>{metrics.duration_seconds:.2f}s</b>) |
                            Final Position Error: <b>{metrics.final_position_error:.3f} m</b> |
                            Wall Time: <b>{metrics.execution_wall_time_ms:.1f} ms</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metric cards
                met1, met2, met3, met4, met5, met6 = st.columns(6)
                met1.metric("RMS Pos Error", f"{metrics.rms_position_error:.3f} m")
                met2.metric("RMS Vel Error", f"{metrics.rms_velocity_error:.3f} m/s")
                met3.metric("Max Attitude Error", f"{metrics.max_attitude_error_deg:.1f}°")
                met4.metric("Mean Angular Speed", f"{metrics.mean_angular_speed:.2f} rad/s")
                met5.metric(
                    "Min Obstacle Clearance",
                    f"{metrics.min_obstacle_clearance:.2f} m"
                    if math.isfinite(metrics.min_obstacle_clearance)
                    else "N/A",
                )
                met6.metric("Control Effort", f"{metrics.control_effort:.1f}")

                # 3D Navigation Scene: Planned vs Flown
                flown_pts = [t.position for t in res.telemetry]
                all_obs = list(selected_sc.obstacles) + list(selected_sc.dynamic_obstacles)
                nav_3d_fig = build_3d_navigation_trajectory_figure(
                    planned_waypoints=selected_sc.waypoints,
                    flown_trajectory=flown_pts,
                    obstacles=all_obs,
                    bounds=selected_sc.bounds,
                    title=f"3D Navigation Trajectory — {res.scenario_name}",
                )
                st.plotly_chart(nav_3d_fig, width="stretch")

                # Actuator Telemetry: Commanded vs Actual Motor Thrust
                act_fig = build_actuator_telemetry_figure(
                    res.telemetry,
                    title=f"Motor Thrust Telemetry (Commanded vs Actual) — {res.scenario_name}",
                )
                st.plotly_chart(act_fig, width="stretch")

                # Synchronized telemetry time-series
                telemetry_fig = build_telemetry_time_series_figure(
                    res.telemetry,
                    title=f"Flight Dynamics Telemetry Time Series — {res.scenario_name}",
                )
                st.plotly_chart(telemetry_fig, width="stretch")

                # Export buttons
                exp_c1, exp_c2 = st.columns(2)
                with exp_c1:
                    json_str = json.dumps(res.to_dict(), indent=2)
                    st.download_button(
                        "📥 Download Scenario JSON Record",
                        data=json_str,
                        file_name=f"{res.scenario_name}_flight_record.json",
                        mime="application/json",
                    )
                with exp_c2:
                    csv_export_path = Path("artifacts") / f"{res.scenario_name}_telemetry.csv"
                    res.export_telemetry_csv(csv_export_path)
                    if csv_export_path.is_file():
                        st.download_button(
                            "📥 Download Telemetry CSV",
                            data=csv_export_path.read_text(encoding="utf-8"),
                            file_name=f"{res.scenario_name}_telemetry.csv",
                            mime="text/csv",
                        )

        else:
            st.markdown("### PPO & SAC Autonomous Flight Benchmarking Matrix")
            st.markdown(
                "Standardized comparative evaluation across **3-DOF Kinematic** and **6-DOF Rigid-Body** environments "
                "with deterministic seed protocols and confidence intervals."
            )

            b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 1.2])
            with b_col1:
                benchmark_profile = st.selectbox(
                    "Execution Profile:",
                    options=["quick", "full"],
                    index=0,
                    help="'quick': Fast validation (2 episodes per combination, smoke testing). 'full': Multi-seed statistical benchmark.",
                )
            with b_col2:
                deterministic_eval = st.checkbox(
                    "Deterministic Policy Actions",
                    value=True,
                    help="Evaluate deterministic mean actions (True) or stochastic distribution actions (False).",
                )
            with b_col3:
                train_missing = st.checkbox(
                    "Train Smoke Baseline if Missing",
                    value=False,
                    help="Train a 512-step smoke policy if no pre-trained checkpoints exist.",
                )

            if st.button("📊 Run Autonomous Flight Benchmark Suite", type="primary"):
                with st.spinner(f"Running flight benchmark suite ({benchmark_profile} profile)..."):
                    suite = FlightBenchmarkSuite()
                    report = suite.run_suite(
                        profile=benchmark_profile,
                        deterministic=deterministic_eval,
                        train_if_missing=train_missing,
                    )
                    st.session_state["benchmark_report"] = report

            if "benchmark_report" in st.session_state:
                rep = st.session_state["benchmark_report"]
                st.markdown(
                    f"""
                    <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 18px; margin: 12px 0;">
                        <span class="arl-pill arl-pill-cyan">● BENCHMARK REPORT: {rep.report_id}</span>
                        <span style="color: #94a3b8; margin-left: 10px; font-size: 13px;">
                            Profile: <b>{rep.profile.upper()}</b> | Combinations Evaluated: <b>{len(rep.combinations_evaluated)}</b> | UTC: <b>{rep.timestamp_utc[:19]}</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Benchmark bar chart
                b_fig = build_flight_benchmark_figure(rep.combinations_evaluated)
                st.plotly_chart(b_fig, width="stretch")

                # Summary Table
                table_rows = []
                for c in rep.combinations_evaluated:
                    ci_str = ""
                    if "reward" in c.confidence_intervals:
                        ci = c.confidence_intervals["reward"]
                        if ci.get("ci95_lower") is not None and ci.get("ci95_upper") is not None:
                            ci_str = f" [{ci['ci95_lower']:.1f}, {ci['ci95_upper']:.1f}]"

                    table_rows.append(
                        {
                            "Algorithm": c.algorithm,
                            "Dynamics Mode": c.mode,
                            "Episodes": c.total_episodes,
                            "Success Rate": f"{c.success_rate * 100:.1f}%",
                            "Collision Rate": f"{c.collision_rate * 100:.1f}%",
                            "Timeout Rate": f"{c.timeout_rate * 100:.1f}%",
                            "Mean Reward": f"{c.mean_reward:.2f}{ci_str}",
                            "Flight Duration": f"{c.mean_flight_duration_seconds:.2f}s",
                            "Runtime": f"{c.evaluation_runtime_seconds:.2f}s",
                        }
                    )
                st.table(table_rows)

                # Export buttons
                b_exp1, b_exp2 = st.columns(2)
                with b_exp1:
                    st.download_button(
                        "📥 Download Benchmark JSON Report",
                        data=json.dumps(rep.to_dict(), indent=2),
                        file_name=f"{rep.report_id}.json",
                        mime="application/json",
                    )
                with b_exp2:
                    csv_path = Path("artifacts") / f"{rep.report_id}.csv"
                    rep.export_csv(csv_path)
                    if csv_path.is_file():
                        st.download_button(
                            "📥 Download Benchmark CSV Summary",
                            data=csv_path.read_text(encoding="utf-8"),
                            file_name=f"{rep.report_id}.csv",
                            mime="text/csv",
                        )


if __name__ == "__main__":
    main()
