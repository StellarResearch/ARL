"""AdaptiveRL: Autonomous 3D Drone Navigation Flight Deck GUI.

A premium aerospace-grade mission-control Streamlit interface for evaluating
autonomous drone flight systems, rigid-body 6-DOF dynamics, 3-DOF kinematic simulation,
16-ray spherical LiDAR sensing, PPO training, and empirical policy benchmarking.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

from adaptive_rl.algorithms.registry import (
    inspect_model_checkpoint,
    load_algorithm_from_pretrained,
)
from adaptive_rl.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig, TrainingConfig
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv, quaternion_to_euler_angles
from adaptive_rl.evaluation.evaluator import (
    Evaluator,
    evaluate_ppo_policy,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.gui import (
    build_arena_3d_figure,
    build_comparison_bar_chart,
    build_density_experiment_figure,
    build_training_curve_figure,
    get_available_models,
    run_drone_simulation_episode,
)
from adaptive_rl.training.trainer import PPOTrainer

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
            "📈 Train PPO",
            "📊 Evaluation & Baseline",
            "🎯 Difficulty Experiment",
            "📘 About & Architecture",
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
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([1.3, 1.8, 0.9, 0.9, 1.0, 1.2])

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
                launch_clicked = st.button(
                    "🚀 Execute Sim", type="primary", use_container_width=True
                )

            if selected_policy.startswith("Random Action Baseline"):
                st.markdown(
                    "<div class='baseline-disclaimer'>⚠️ <b>Untrained Baseline Note:</b> Random Action Baseline provides "
                    "stochastic control inputs for baseline comparison; it does not represent a trained agent.</div>",
                    unsafe_allow_html=True,
                )

        is_6dof = "6-DOF" in env_choice
        sim_key = f"flight_data_{'6dof' if is_6dof else '3dof'}"

        if launch_clicked or sim_key not in st.session_state:
            with st.spinner(f"Executing {env_choice} numerical dynamics simulation..."):
                env: Any = None
                if is_6dof:
                    env = Drone6DOFEnv(num_obstacles=int(num_obs))
                else:
                    env = DroneNavigation3DEnv(num_obstacles=int(num_obs))

                try:
                    model_to_use: Optional[Any] = None

                    if not selected_policy.startswith("Random Action Baseline") and available_models:
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

                    st.session_state[sim_key] = run_drone_simulation_episode(
                        env=env,
                        model=model_to_use,
                        seed=int(demo_seed),
                    )
                finally:
                    env.close()

        flight_data = st.session_state[sim_key]
        steps = flight_data["steps"]
        total_steps = len(steps) - 1

        # Flight Outcome Status Banner
        outcome_status = flight_data["outcome"]
        is_success = outcome_status == "SUCCESS"
        banner_cls = "success" if is_success else "failure"
        badge_cls = "arl-pill-green" if is_success else "arl-pill-red"
        outcome_banner_text = "MISSION SUCCESSFUL" if is_success else "COLLISION / TIMEOUT"

        final_dist = steps[-1]["distance_to_goal"]
        st.markdown(
            f"""
            <div class="arl-outcome-banner {banner_cls}">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span class="arl-pill {badge_cls}">● {outcome_banner_text}</span>
                        <span style="font-weight: 700; font-size: 15px; color: #f1f5f9;">{env_choice}</span>
                        <span style="color: #64748b;">|</span>
                        <span style="color: #94a3b8; font-size: 13px;">Policy: <b>{selected_policy.split(': ')[0]}</b></span>
                    </div>
                    <div style="margin-top: 4px; font-size: 13px; color: #cbd5e1;">
                        Episode finished in <b>{flight_data["total_steps"]}</b> timesteps with cumulative reward <b>{flight_data["total_reward"]:.2f}</b> (Seed: {flight_data["seed"]}).
                    </div>
                </div>
                <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #94a3b8;">
                    Final Goal Dist: <b style="color: #f8fafc;">{final_dist:.2f} m</b><br>
                    Physics Mode: <b style="color: #38bdf8;">{'RK4 6-DOF' if is_6dof else 'Kinematic 3-DOF'}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Scrubber & Viewport Control Bar
        vp_col1, vp_col2, vp_col3, vp_col4, vp_col5, vp_col6 = st.columns([2.4, 0.7, 1.4, 0.7, 0.7, 0.7])
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
                options=["Arena Overview", "Drone Follow", "Drone Inspection", "Target View", "Free Orbit", "Reset Camera"],
                index=0,
                help="Select 3D viewport camera perspective. Use 'Drone Inspection' for detailed close-up examination of quadrotor airframe.",
            )
        with vp_col4:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            show_floor = st.checkbox("Floor", value=True, help="Render 3D flight test chamber floor and grid.")
        with vp_col5:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            show_safety = st.checkbox("Safety", value=True, help="Render obstacle safety proximity zones.")
        with vp_col6:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            show_traj = st.checkbox("Path", value=True, help="Render recorded 3D flight trajectory.")

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
            )
            # Give viewport substantial height for dominance
            fig.update_layout(height=660)
            st.plotly_chart(fig, use_container_width=True)

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
                            <span class="telemetry-stat-val green">{step_info['distance_to_goal']:.2f}<span class="unit">m</span></span>
                        </div>
                        <div class="telemetry-stat">
                            <span class="telemetry-stat-label">Nearest Obstacle</span>
                            <span class="telemetry-stat-val {'red' if step_info['min_obstacle_distance'] < 1.0 else 'amber'}">{step_info['min_obstacle_distance']:.2f}<span class="unit">m</span></span>
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
                            <span class="telemetry-stat-val cyan">{step_info['speed']:.2f}<span class="unit">m/s</span></span>
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
                            <span class="telemetry-stat-val {'red' if min_lidar < 1.5 else 'green'}">{min_lidar:.2f}<span class="unit">m</span></span>
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
                            <span class="telemetry-stat-val">{step_info['reward']:+.2f}</span>
                        </div>
                        <div class="telemetry-stat">
                            <span class="telemetry-stat-label">Cumulative Return</span>
                            <span class="telemetry-stat-val green">{step_info['cumulative_reward']:+.2f}</span>
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

    # ==========================================
    # TAB 2: TRAIN PPO
    # ==========================================
    with tabs[1]:
        st.subheader("Train PPO Policy")
        st.markdown(
            "Execute real reinforcement learning training using **Stable-Baselines3 PPO** "
            "on your CPU. Training progress and rewards are tracked in real-time."
        )

        train_col1, train_col2, train_col3 = st.columns(3)
        with train_col1:
            train_steps = st.select_slider(
                "Training Budget (Timesteps):",
                options=[5000, 10000, 25000, 50000],
                value=25000,
                help="25,000 steps requires ~25 seconds on a standard student laptop CPU.",
            )
        with train_col2:
            train_lr = st.selectbox(
                "Learning Rate:",
                options=[0.0001, 0.0003, 0.001],
                index=1,
            )
        with train_col3:
            train_seed = st.number_input("Random Seed:", value=42, step=1)

        is_training = st.session_state.get("is_training", False)
        start_train_btn = st.button("▶ Start PPO Training", type="primary", disabled=is_training)

        if start_train_btn:
            st.session_state["is_training"] = True
            st.info(f"Starting training run ({train_steps:,} timesteps on CPU)...")
            prog_bar = st.progress(0.0)
            train_status_box: Any = st.empty()

            try:
                train_cfg = ExperimentConfig(
                    name="drone_ppo_gui",
                    seed=int(train_seed),
                    algorithm=AlgorithmConfig(
                        name="ppo",
                        learning_rate=float(train_lr),
                    ),
                    environment=EnvironmentConfig(
                        name="drone",
                    ),
                    training=TrainingConfig(
                        total_timesteps=int(train_steps),
                    ),
                    output_dir=Path("artifacts"),
                )

                trainer = PPOTrainer(config=train_cfg)
                t0 = time.time()
                result = trainer.fit()
                duration = time.time() - t0

                prog_bar.progress(1.0)
                train_status_box.text(f"Training Complete! Duration: {duration:.2f}s")
                st.success(f"Checkpoint saved: `{result.final_model_path}`")

                st.session_state["latest_training_rewards"] = result.episode_rewards

                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Timesteps", f"{result.total_timesteps:,}")
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
            st.plotly_chart(fig_curve, use_container_width=True)

    # ==========================================
    # TAB 3: EVALUATION & BASELINE
    # ==========================================
    with tabs[2]:
        st.subheader("Policy Evaluation & Random Baseline Comparison")
        st.markdown(
            "Empirically compare the trained **PPO agent** against an **untrained Random Policy baseline** "
            "under identical environment seeds and episode counts."
        )

        e_col1, e_col2, e_col3 = st.columns([2, 1, 1])
        with e_col1:
            eval_model_name = st.selectbox(
                "Evaluation Model:",
                options=model_names if model_names else ["No models found in artifacts/models"],
                index=0,
            )
        with e_col2:
            eval_episodes = st.slider("Episodes:", min_value=5, max_value=50, value=20, step=5)
        with e_col3:
            eval_mode = st.radio(
                "Mode:", ["Compare Both (PPO vs Random)", "PPO Only", "Random Only"]
            )

        run_eval_btn = st.button("📊 Run Benchmark Evaluation", type="primary")

        if run_eval_btn:
            with st.spinner(f"Running evaluation ({eval_episodes} episodes)..."):
                eval_env = DroneNavigation3DEnv()
                try:
                    metrics_to_plot: Dict[str, Any] = {}

                    # Evaluate Model
                    if eval_mode in ["Compare Both (PPO vs Random)", "PPO Only"]:
                        if available_models and eval_model_name in model_info_map:
                            target_path = next(
                                p for p in available_models if p.name == eval_model_name
                            )
                            info = model_info_map[target_path.name]
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
                                    f"and action shape {info['action_shape']}, but 3-DOF benchmark environment requires "
                                    f"{eval_env.observation_space.shape} and {eval_env.action_space.shape}. "
                                    "Cannot evaluate incompatible model."
                                )
                            else:
                                trained_algo = load_algorithm_from_pretrained(
                                    target_path, env=eval_env
                                )
                                trained_res = evaluate_ppo_policy(
                                    algorithm=trained_algo,
                                    env=eval_env,
                                    num_episodes=int(eval_episodes),
                                    base_seed=42,
                                )
                                algo_title = (
                                    f"{info['algorithm'].upper()} Policy"
                                    if info["algorithm"]
                                    else "Trained Policy"
                                )
                                metrics_to_plot[algo_title] = trained_res
                                Evaluator.save_report(trained_res, Path("artifacts/evaluation.json"))
                        else:
                            st.warning("No saved model found. Please train a compatible agent first.")

                    # Evaluate Random Policy
                    if eval_mode in ["Compare Both (PPO vs Random)", "Random Only"]:
                        rand_res = evaluate_random_policy(
                            env=eval_env,
                            num_episodes=int(eval_episodes),
                            base_seed=42,
                        )
                        metrics_to_plot["Random Policy"] = rand_res

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
            st.plotly_chart(fig_comp, use_container_width=True)
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
            st.plotly_chart(fig_density, use_container_width=True)
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


if __name__ == "__main__":
    main()
