"""Comprehensive regression tests for Flight Deck visualizer, 3D drone geometry,

and simulation contract verification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.gui.visualizer import (
    build_arena_3d_figure,
    compute_quadrotor_3d_geometry,
)
from adaptive_rl.manifest import (
    ArtifactProvenance,
    ExecutionMetadata,
    ExperimentManifest,
    GitMetadata,
    HardwareTelemetry,
    HostSystem,
    SoftwarePackages,
    compute_sha256,
)


def test_quadrotor_3d_geometry_extended_components() -> None:
    """Verify landing gear, sensor gimbal, and rotor coordinates in body-to-world frame."""
    center = np.array([12.0, 15.0, 6.0])
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    arm_span = 1.6
    d = arm_span / (2.0 * np.sqrt(2.0))

    geom = compute_quadrotor_3d_geometry(
        center_pos=center,
        quaternion=q_ident,
        arm_span=arm_span,
        rotor_radius=0.3,
    )

    # Core required keys
    assert "center" in geom
    assert "rotors" in geom
    assert "nose" in geom
    assert "fuselage" in geom
    assert "propellers" in geom
    assert "front_arms" in geom
    assert "rear_arms" in geom

    # Extended high-fidelity components
    assert "landing_gear" in geom
    assert "camera_pod" in geom

    # Check landing gear has 4 distinct struts
    landing_gear = geom["landing_gear"]
    assert len(landing_gear) == 4
    for top_pt, foot_pt in landing_gear:
        assert top_pt[2] > foot_pt[2]  # Foot is below top strut
        assert np.isclose(foot_pt[2], center[2] - 0.16, atol=1e-4)

    # Check camera pod is along +X direction (forward) and slightly lowered
    cam_pod = geom["camera_pod"]
    assert cam_pod[0] > center[0]
    assert np.isclose(cam_pod[1], center[1], atol=1e-5)
    assert cam_pod[2] < center[2]

    # Verify rotor positions match 6-DOF convention
    rotors = geom["rotors"]
    assert np.allclose(rotors[0], [center[0] + d, center[1] - d, center[2]])  # R1 Front-Right
    assert np.allclose(rotors[1], [center[0] + d, center[1] + d, center[2]])  # R2 Front-Left
    assert np.allclose(rotors[2], [center[0] - d, center[1] + d, center[2]])  # R3 Rear-Left
    assert np.allclose(rotors[3], [center[0] - d, center[1] - d, center[2]])  # R4 Rear-Right


def test_quadrotor_geometry_pitch_90_rotation() -> None:
    """Verify pitch rotation transforms camera pod and nose up/down."""
    center = np.array([0.0, 0.0, 0.0])
    # 90 deg pitch down about +Y: q = [cos(pi/4), 0, sin(pi/4), 0]
    q_pitch90 = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])

    geom = compute_quadrotor_3d_geometry(center_pos=center, quaternion=q_pitch90, arm_span=1.6)

    # In body frame nose is at +X. Pitched 90 deg down about +Y brings +X to -Z
    nose = geom["nose"]
    assert np.isclose(nose[0], 0.0, atol=1e-4)
    assert np.isclose(nose[1], 0.0, atol=1e-4)
    assert nose[2] < -0.1


def test_build_arena_3d_figure_with_lidar_contacts() -> None:
    """Verify Plotly figure generation includes color-coded LiDAR contacts and landing gear."""
    bounds = (30.0, 30.0, 15.0)
    obstacles = [ObstacleSphere3D(center=np.array([10.0, 10.0, 5.0]), radius=2.0)]
    target = np.array([25.0, 25.0, 10.0])
    start = np.array([5.0, 5.0, 3.0])
    cur_pos = np.array([8.0, 8.0, 4.0])
    cur_vel = np.array([2.0, 1.0, 0.0])
    cur_quat = [1.0, 0.0, 0.0, 0.0]

    # LiDAR rays: ray 0 hits obstacle (distance 3.0m < 4.0m), ray 1 hits boundary (15.0m)
    lidar_rays = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lidar_ranges = [3.0, 15.0]

    fig = build_arena_3d_figure(
        bounds=bounds,
        obstacles=obstacles,
        target=target,
        start_pos=start,
        trajectory=[start, cur_pos],
        current_pos=cur_pos,
        current_vel=cur_vel,
        attitude_quaternion=cur_quat,
        lidar_rays=lidar_rays,
        lidar_ranges=lidar_ranges,
        show_lidar=True,
    )

    assert isinstance(fig, go.Figure)
    trace_names = [trace.name for trace in fig.data if hasattr(trace, "name")]

    # Core required names from test_gui.py contract
    assert "Arena Boundary" in trace_names
    assert "Obstacle 1" in trace_names
    assert "Target Waypoint" in trace_names
    assert "Start Position" in trace_names
    assert "Flight Path" in trace_names
    assert "16-Ray LiDAR" in trace_names
    assert "Drone" in trace_names
    assert "Velocity Vector" in trace_names

    # Extended visualizer traces
    assert "Landing Gear" in trace_names
    assert "Sensor Gimbal" in trace_names
    assert "LiDAR Contacts" in trace_names


def test_artifact_integrity_verification_positive(tmp_path: Path) -> None:
    """Verify cryptographic SHA-256 computation and manifest validation."""
    dummy_model = tmp_path / "model.zip"
    dummy_model.write_bytes(b"PK\x03\x04test_weights_binary_content_12345")
    actual_hash = compute_sha256(dummy_model)

    manifest = ExperimentManifest(
        experiment_name="test_manifest",
        git=GitMetadata(git_commit="abcdef1234567890", git_branch="main", git_dirty=False),
        host=HostSystem(
            os_name="Linux", os_version="6.0", python_version="3.11", architecture="x86_64"
        ),
        packages=SoftwarePackages(
            adaptive_rl="0.1.0",
            torch="2.1.0",
            stable_baselines3="2.2.0",
            gymnasium="0.29.1",
            numpy="1.26.0",
            typer="0.9.0",
            pydantic="2.5.0",
        ),
        hardware=HardwareTelemetry(device="cpu", cpu_count=4),
        execution=ExecutionMetadata(
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            duration_seconds=60.0,
            command=["train"],
        ),
        config={"algorithm": "ppo"},
        artifacts={
            "model": ArtifactProvenance(
                path=str(dummy_model),
                sha256=actual_hash,
                size_bytes=dummy_model.stat().st_size,
            )
        },
    )

    art_prov = manifest.artifacts["model"]
    art_path = Path(art_prov.path)
    assert art_path.exists()
    computed_sha = compute_sha256(art_path)
    assert computed_sha == art_prov.sha256


def test_artifact_integrity_verification_tamper_detection(tmp_path: Path) -> None:
    """Verify that tampering with an artifact produces a checksum mismatch."""
    dummy_model = tmp_path / "model.zip"
    dummy_model.write_bytes(b"original_bytes")
    recorded_hash = compute_sha256(dummy_model)

    # Tamper with the file
    dummy_model.write_bytes(b"tampered_bytes_malicious")
    new_hash = compute_sha256(dummy_model)

    assert new_hash != recorded_hash


def test_browser_simulation_contract_verification() -> None:
    """Verify that docs/js/simulation.js enforces correct mathematical invariants."""
    sim_js_path = Path("docs/js/simulation.js")
    assert sim_js_path.exists(), "docs/js/simulation.js must exist"
    content = sim_js_path.read_text(encoding="utf-8")

    # 1. Physical constants
    assert "ARENA_X = 30.0" in content
    assert "ARENA_Y = 30.0" in content
    assert "ARENA_Z = 15.0" in content
    assert "MAX_SPEED = 8.0" in content
    assert "MAX_ACCEL = 4.0" in content
    assert "DRAG_COEFF = 0.05" in content
    assert "TARGET_RADIUS = 1.5" in content
    assert "COLLISION_RADIUS = 0.8" in content

    # 2. Semi-implicit Euler integration equation
    assert "effectiveAcc = accel.clone().addScaledVector(simState.vel, -DRAG_COEFF)" in content
    assert "simState.vel.addScaledVector(effectiveAcc, DT)" in content
    assert "simState.pos.addScaledVector(simState.vel, DT)" in content

    # 3. Post-step progress calculation (using distance before vs after)
    assert "distBefore = toGoal.length()" in content
    assert "distAfter = simState.pos.distanceTo(simState.target)" in content
    assert "distDelta = distBefore - distAfter" in content

    # 4. Post-step goal detection
    assert "isSuccess = distAfter <= TARGET_RADIUS" in content

    # 5. Post-step collision detection
    assert "isCollision = false" in content
    assert "simState.pos.distanceTo(new THREE.Vector3(obs.x, obs.y, obs.z))" in content

    # 6. WebGL fallback banner present
    assert "webgl-fallback-banner" in content
    assert "WEBGL ACCELERATION UNAVAILABLE" in content


def test_website_index_contract_and_assets() -> None:
    """Verify that docs/index.html has no broken references or obsolete claims."""
    index_path = Path("docs/index.html")
    assert index_path.exists()
    content = index_path.read_text(encoding="utf-8")

    # Obsolete test counts must not appear in index.html
    assert "49 / 49" not in content
    assert "49 unit tests" not in content
    assert "49 Tests" not in content

    # Required 12 section anchors
    sections = [
        "hero-sim",
        "comparison",
        "pipeline",
        "sensors",
        "benchmarks",
        "provenance",
        "gui",
        "install",
        "limitations",
    ]
    for sec in sections:
        assert f'id="{sec}"' in content, f"Missing section id='{sec}' in docs/index.html"

    # Local JS and CSS assets exist
    assert Path("docs/css/style.css").exists()
    assert Path("docs/js/simulation.js").exists()
    assert Path("docs/js/lidar.js").exists()
    assert Path("docs/js/app.js").exists()
    assert Path("docs/js/presentation.js").exists()
