"""Plotly-based 3D visual arena and sensor visualizer for AdaptiveRL.

Provides premium aerospace-grade 3D mesh rendering of autonomous quadrotors,
spherical obstacle safety fields, LiDAR sensor rays, and mission-control charts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from adaptive_rl.algorithms.registry import inspect_model_checkpoint
from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import (
    quaternion_to_rotation_matrix,
)


def _combine_meshes(
    mesh_list: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine multiple (x, y, z, i, j, k) mesh tuples into a single unified mesh."""
    xs, ys, zs = [], [], []
    is_, js_, ks_ = [], [], []
    offset = 0
    for x, y, z, i, j, k in mesh_list:
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
        zs.append(z)
        is_.append(i + offset)
        js_.append(j + offset)
        ks_.append(k + offset)
        offset += len(x)
    if not xs:
        empty = np.array([], dtype=np.float64)
        empty_int = np.array([], dtype=np.int32)
        return empty, empty, empty, empty_int, empty_int, empty_int
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(zs),
        np.concatenate(is_),
        np.concatenate(js_),
        np.concatenate(ks_),
    )


def _cylinder_mesh_body(
    p_start: np.ndarray,
    p_end: np.ndarray,
    radius: float,
    n_seg: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3D cylinder mesh between p_start and p_end in body coordinates."""
    v = p_end - p_start
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        empty = np.array([], dtype=np.float64)
        empty_int = np.array([], dtype=np.int32)
        return empty, empty, empty, empty_int, empty_int, empty_int
    v_n = v / length
    if abs(v_n[0]) < 0.8:
        arb = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        arb = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u1 = np.cross(v_n, arb)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(v_n, u1)

    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    pts_s = [p_start + radius * (math.cos(th) * u1 + math.sin(th) * u2) for th in theta]
    pts_e = [p_end + radius * (math.cos(th) * u1 + math.sin(th) * u2) for th in theta]
    verts = np.array(pts_s + pts_e + [p_start, p_end], dtype=np.float64)

    i_l, j_l, k_l = [], [], []
    c_s = 2 * n_seg
    c_e = 2 * n_seg + 1
    for s in range(n_seg):
        s_nxt = (s + 1) % n_seg
        i_l.extend([s, s])
        j_l.extend([s_nxt, n_seg + s_nxt])
        k_l.extend([n_seg + s_nxt, n_seg + s])
        i_l.append(c_s)
        j_l.append(s_nxt)
        k_l.append(s)
        i_l.append(c_e)
        j_l.append(n_seg + s)
        k_l.append(n_seg + s_nxt)

    return (
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        np.array(i_l, dtype=np.int32),
        np.array(j_l, dtype=np.int32),
        np.array(k_l, dtype=np.int32),
    )


def _sphere_mesh_body(
    center: np.ndarray,
    radius: float,
    n_lat: int = 8,
    n_lon: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3D sphere mesh in body coordinates."""
    verts = []
    verts.append(center + np.array([0.0, 0.0, radius]))
    for lat in range(1, n_lat):
        phi = math.pi * lat / n_lat
        z = radius * math.cos(phi)
        r_xy = radius * math.sin(phi)
        for lon in range(n_lon):
            theta = 2 * math.pi * lon / n_lon
            x = r_xy * math.cos(theta)
            y = r_xy * math.sin(theta)
            verts.append(center + np.array([x, y, z]))
    verts.append(center + np.array([0.0, 0.0, -radius]))
    verts_arr = np.array(verts, dtype=np.float64)

    i_l, j_l, k_l = [], [], []
    for lon in range(n_lon):
        nxt = (lon + 1) % n_lon
        i_l.append(0)
        j_l.append(1 + lon)
        k_l.append(1 + nxt)

    for lat in range(n_lat - 2):
        row1 = 1 + lat * n_lon
        row2 = 1 + (lat + 1) * n_lon
        for lon in range(n_lon):
            nxt = (lon + 1) % n_lon
            i_l.extend([row1 + lon, row1 + lon])
            j_l.extend([row1 + nxt, row2 + nxt])
            k_l.extend([row2 + nxt, row2 + lon])

    south_idx = len(verts) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for lon in range(n_lon):
        nxt = (lon + 1) % n_lon
        i_l.append(south_idx)
        j_l.append(last_row + nxt)
        k_l.append(last_row + lon)

    return (
        verts_arr[:, 0],
        verts_arr[:, 1],
        verts_arr[:, 2],
        np.array(i_l, dtype=np.int32),
        np.array(j_l, dtype=np.int32),
        np.array(k_l, dtype=np.int32),
    )


def _blade_mesh_body(
    rotor_center: np.ndarray,
    blade_length: float,
    chord: float,
    angle_rad: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2-blade aerodynamic propeller mesh in body coordinates."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dir_v = np.array([cos_a, sin_a, 0.0], dtype=np.float64)
    nor_v = np.array([-sin_a, cos_a, 0.0], dtype=np.float64)

    z = rotor_center[2] + 0.04
    p0 = rotor_center + dir_v * 0.04 - nor_v * (chord * 0.5)
    p1 = rotor_center + dir_v * 0.04 + nor_v * (chord * 0.5)
    p2 = rotor_center + dir_v * blade_length + nor_v * (chord * 0.28)
    p3 = rotor_center + dir_v * blade_length - nor_v * (chord * 0.28)

    p4 = rotor_center - dir_v * 0.04 - nor_v * (chord * 0.5)
    p5 = rotor_center - dir_v * 0.04 + nor_v * (chord * 0.5)
    p6 = rotor_center - dir_v * blade_length + nor_v * (chord * 0.28)
    p7 = rotor_center - dir_v * blade_length - nor_v * (chord * 0.28)

    verts = np.array([
        [p0[0], p0[1], z],
        [p1[0], p1[1], z],
        [p2[0], p2[1], z + 0.01],
        [p3[0], p3[1], z + 0.01],
        [p4[0], p4[1], z],
        [p5[0], p5[1], z],
        [p6[0], p6[1], z - 0.01],
        [p7[0], p7[1], z - 0.01],
    ], dtype=np.float64)

    i = np.array([0, 0, 4, 4], dtype=np.int32)
    j = np.array([1, 2, 5, 6], dtype=np.int32)
    k = np.array([2, 3, 6, 7], dtype=np.int32)
    return verts[:, 0], verts[:, 1], verts[:, 2], i, j, k


def _fuselage_mesh_body(d: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sleek 3D faceted aerospace fuselage pod in body coordinates."""
    top_pts = [
        [0.38 * d, 0.0, 0.02 * d],        # 0: nose
        [0.22 * d, 0.18 * d, 0.05 * d],   # 1: front-left
        [0.0, 0.22 * d, 0.05 * d],        # 2: mid-left
        [-0.22 * d, 0.16 * d, 0.04 * d],  # 3: rear-left
        [-0.34 * d, 0.0, 0.02 * d],       # 4: tail
        [-0.22 * d, -0.16 * d, 0.04 * d], # 5: rear-right
        [0.0, -0.22 * d, 0.05 * d],       # 6: mid-right
        [0.22 * d, -0.18 * d, 0.05 * d],  # 7: front-right
    ]
    bot_pts = [
        [0.34 * d, 0.0, -0.03 * d],
        [0.20 * d, 0.16 * d, -0.06 * d],
        [0.0, 0.20 * d, -0.06 * d],
        [-0.20 * d, 0.14 * d, -0.05 * d],
        [-0.30 * d, 0.0, -0.03 * d],
        [-0.20 * d, -0.14 * d, -0.05 * d],
        [0.0, -0.20 * d, -0.06 * d],
        [0.20 * d, -0.16 * d, -0.06 * d],
    ]
    crest = [0.05 * d, 0.0, 0.09 * d]
    keel = [0.0, 0.0, -0.08 * d]

    verts = np.array(top_pts + bot_pts + [crest, keel], dtype=np.float64)
    c_idx = 16
    k_idx = 17

    i_l, j_l, k_l = [], [], []
    for s in range(8):
        nxt = (s + 1) % 8
        i_l.append(c_idx)
        j_l.append(s)
        k_l.append(nxt)

    for s in range(8):
        nxt = (s + 1) % 8
        i_l.append(k_idx)
        j_l.append(8 + nxt)
        k_l.append(8 + s)

    for s in range(8):
        nxt = (s + 1) % 8
        top_s, top_nxt = s, nxt
        bot_s, bot_nxt = 8 + s, 8 + nxt
        i_l.extend([top_s, top_s])
        j_l.extend([bot_s, top_nxt])
        k_l.extend([bot_nxt, bot_nxt])

    return (
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        np.array(i_l, dtype=np.int32),
        np.array(j_l, dtype=np.int32),
        np.array(k_l, dtype=np.int32),
    )


def _chevron_mesh_body(d: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate forward heading indicator chevron in body coordinates."""
    z0 = 0.03 * d
    z1 = 0.06 * d
    pts_2d = [
        [0.65 * d, 0.0],
        [0.44 * d, 0.12 * d],
        [0.50 * d, 0.0],
        [0.44 * d, -0.12 * d],
    ]
    verts_lower = [[p[0], p[1], z0] for p in pts_2d]
    verts_upper = [[p[0], p[1], z1] for p in pts_2d]
    verts = np.array(verts_lower + verts_upper, dtype=np.float64)

    i = np.array([4, 4, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    j = np.array([5, 6, 2, 3, 1, 5, 2, 6, 3, 7, 0, 4], dtype=np.int32)
    k = np.array([6, 7, 1, 2, 5, 4, 6, 5, 7, 6, 4, 7], dtype=np.int32)
    return verts[:, 0], verts[:, 1], verts[:, 2], i, j, k


def _transform_mesh(
    mesh_b: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    center_pos: np.ndarray,
    rot_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Transform body mesh (xb, yb, zb, i, j, k) to world coordinates."""
    xb, yb, zb, i, j, k = mesh_b
    if len(xb) == 0:
        empty = np.array([], dtype=np.float64)
        empty_int = np.array([], dtype=np.int32)
        return {"x": empty, "y": empty, "z": empty, "i": empty_int, "j": empty_int, "k": empty_int}
    v_b = np.column_stack([xb, yb, zb])
    v_w = center_pos + (rot_mat @ v_b.T).T
    return {
        "x": v_w[:, 0],
        "y": v_w[:, 1],
        "z": v_w[:, 2],
        "i": i,
        "j": j,
        "k": k,
    }


def compute_quadrotor_3d_geometry(
    center_pos: np.ndarray | Sequence[float],
    quaternion: Optional[np.ndarray | Sequence[float]] = None,
    arm_span: float = 1.2,
    rotor_radius: float = 0.28,
    n_propeller_points: int = 16,
) -> Dict[str, Any]:
    """Compute 3D world-coordinate geometry for an X-configuration quadrotor.

    Geometry components:
    - Central body fuselage with 3D volume pod and forward heading nose (+X_B direction)
    - 4 volumetric X-configuration arms with structural motor housings
    - 4 rotor hubs (R1: Front-Right, R2: Front-Left, R3: Rear-Left, R4: Rear-Right)
    - 4 recognizable 3D propellers with front/rear color differentiation
    - Dual port/starboard landing gear skids and angled structural struts
    - Forward-facing optical sensor gimbal and camera assembly
    - Illuminated 3D navigation chevron indicating heading (+X_B)

    Transformation:
      P_W = center_pos + R(q) @ P_B
    """
    c_pos = np.asarray(center_pos, dtype=np.float64)
    if c_pos.shape != (3,):
        raise ValueError(f"center_pos must have shape (3,), got {c_pos.shape}")
    if not np.all(np.isfinite(c_pos)):
        raise ValueError(f"center_pos must have finite coordinates, got {center_pos}")

    if quaternion is not None:
        q_arr = np.asarray(quaternion, dtype=np.float64)
        if q_arr.shape != (4,):
            raise ValueError(f"quaternion must have shape (4,), got {q_arr.shape}")
        if not np.all(np.isfinite(q_arr)):
            raise ValueError(f"quaternion must have finite coordinates, got {quaternion}")
        rot_mat = quaternion_to_rotation_matrix(q_arr)
    else:
        rot_mat = np.eye(3, dtype=np.float64)

    # Arm offset in local body frame (+X forward, +Y left, +Z up)
    # Rotor 1: (+d, -d, 0) [Front-Right]
    # Rotor 2: (+d, +d, 0) [Front-Left]
    # Rotor 3: (-d, +d, 0) [Rear-Left]
    # Rotor 4: (-d, -d, 0) [Rear-Right]
    d = float(arm_span) / (2.0 * math.sqrt(2.0))

    body_rotors = [
        np.array([d, -d, 0.0], dtype=np.float64),  # R1 Front-Right
        np.array([d, d, 0.0], dtype=np.float64),   # R2 Front-Left
        np.array([-d, d, 0.0], dtype=np.float64),  # R3 Rear-Left
        np.array([-d, -d, 0.0], dtype=np.float64), # R4 Rear-Right
    ]

    # Transform rotors to world coordinates
    world_rotors = [c_pos + rot_mat @ r_b for r_b in body_rotors]

    # Forward heading indicator: nose pointer along +X_B
    nose_body = np.array([d * 0.95, 0.0, 0.0], dtype=np.float64)
    nose_world = c_pos + rot_mat @ nose_body

    # Central fuselage ring / diamond in body frame
    fuselage_pts_b = [
        np.array([d * 0.35, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, d * 0.35, 0.0], dtype=np.float64),
        np.array([-d * 0.35, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, -d * 0.35, 0.0], dtype=np.float64),
        np.array([d * 0.35, 0.0, 0.0], dtype=np.float64),
    ]
    fuselage_pts_w = [c_pos + rot_mat @ p for p in fuselage_pts_b]

    # Propeller rings in body frame (circles in XY plane at z = +0.03)
    theta = np.linspace(0.0, 2.0 * math.pi, n_propeller_points)
    propellers_w: List[np.ndarray] = []
    for r_b in body_rotors:
        ring_pts_b = np.column_stack(
            [
                r_b[0] + rotor_radius * np.cos(theta),
                r_b[1] + rotor_radius * np.sin(theta),
                np.full_like(theta, r_b[2] + 0.03),
            ]
        )
        ring_pts_w = np.array([c_pos + rot_mat @ pt for pt in ring_pts_b])
        propellers_w.append(ring_pts_w)

    # ==========================================
    # DETAILED 3D MESH GEOMETRY COMPONENTS
    # ==========================================
    # 1. 3D Fuselage Pod
    fuse_mesh_b = _fuselage_mesh_body(d)
    mesh_fuselage = _transform_mesh(fuse_mesh_b, c_pos, rot_mat)

    # 2. Four Volumetric 3D Arms (Front Arms and Rear Arms)
    arm_front_cylinders = [
        _cylinder_mesh_body(np.array([0.0, 0.0, 0.0]), body_rotors[0], 0.04 * d),
        _cylinder_mesh_body(np.array([0.0, 0.0, 0.0]), body_rotors[1], 0.04 * d),
    ]
    arm_rear_cylinders = [
        _cylinder_mesh_body(np.array([0.0, 0.0, 0.0]), body_rotors[2], 0.04 * d),
        _cylinder_mesh_body(np.array([0.0, 0.0, 0.0]), body_rotors[3], 0.04 * d),
    ]
    mesh_front_arms = _transform_mesh(_combine_meshes(arm_front_cylinders), c_pos, rot_mat)
    mesh_rear_arms = _transform_mesh(_combine_meshes(arm_rear_cylinders), c_pos, rot_mat)
    mesh_arms = _transform_mesh(_combine_meshes(arm_front_cylinders + arm_rear_cylinders), c_pos, rot_mat)

    # 3. Four Motor Housings
    motors_front = [
        _cylinder_mesh_body(body_rotors[0] + np.array([0, 0, -0.07 * d]), body_rotors[0] + np.array([0, 0, 0.03 * d]), 0.15 * d),
        _cylinder_mesh_body(body_rotors[1] + np.array([0, 0, -0.07 * d]), body_rotors[1] + np.array([0, 0, 0.03 * d]), 0.15 * d),
    ]
    motors_rear = [
        _cylinder_mesh_body(body_rotors[2] + np.array([0, 0, -0.07 * d]), body_rotors[2] + np.array([0, 0, 0.03 * d]), 0.15 * d),
        _cylinder_mesh_body(body_rotors[3] + np.array([0, 0, -0.07 * d]), body_rotors[3] + np.array([0, 0, 0.03 * d]), 0.15 * d),
    ]
    mesh_motors_front = _transform_mesh(_combine_meshes(motors_front), c_pos, rot_mat)
    mesh_motors_rear = _transform_mesh(_combine_meshes(motors_rear), c_pos, rot_mat)

    # 4. Four Rotor Hub Spinners and Assembly
    hubs = [
        _cylinder_mesh_body(r + np.array([0, 0, 0.03 * d]), r + np.array([0, 0, 0.08 * d]), 0.07 * d)
        for r in body_rotors
    ]
    mesh_rotor_hubs = _transform_mesh(_combine_meshes(hubs), c_pos, rot_mat)
    mesh_rotor_assemblies = _transform_mesh(_combine_meshes(motors_front + motors_rear + hubs), c_pos, rot_mat)

    # 5. Four Recognizable Propeller Blades
    blade_len = float(rotor_radius)
    blade_chord = max(0.05 * d, 0.05)
    props_front = [
        _blade_mesh_body(body_rotors[0], blade_len, blade_chord, 0.35),
        _blade_mesh_body(body_rotors[1], blade_len, blade_chord, -0.35),
    ]
    props_rear = [
        _blade_mesh_body(body_rotors[2], blade_len, blade_chord, 0.35),
        _blade_mesh_body(body_rotors[3], blade_len, blade_chord, -0.35),
    ]
    mesh_props_front = _transform_mesh(_combine_meshes(props_front), c_pos, rot_mat)
    mesh_props_rear = _transform_mesh(_combine_meshes(props_rear), c_pos, rot_mat)

    # 6. Landing Gear (Dual Skids and 4 Angled Struts)
    skid_port = [
        _cylinder_mesh_body(np.array([-0.36 * d, 0.22 * d, -0.16 * d]), np.array([0.36 * d, 0.22 * d, -0.16 * d]), 0.02 * d),
        _cylinder_mesh_body(np.array([0.36 * d, 0.22 * d, -0.16 * d]), np.array([0.44 * d, 0.22 * d, -0.08 * d]), 0.02 * d),
        _cylinder_mesh_body(np.array([-0.36 * d, 0.22 * d, -0.16 * d]), np.array([-0.44 * d, 0.22 * d, -0.08 * d]), 0.02 * d),
    ]
    skid_stbd = [
        _cylinder_mesh_body(np.array([-0.36 * d, -0.22 * d, -0.16 * d]), np.array([0.36 * d, -0.22 * d, -0.16 * d]), 0.02 * d),
        _cylinder_mesh_body(np.array([0.36 * d, -0.22 * d, -0.16 * d]), np.array([0.44 * d, -0.22 * d, -0.08 * d]), 0.02 * d),
        _cylinder_mesh_body(np.array([-0.36 * d, -0.22 * d, -0.16 * d]), np.array([-0.44 * d, -0.22 * d, -0.08 * d]), 0.02 * d),
    ]
    struts = [
        _cylinder_mesh_body(np.array([0.16 * d, 0.14 * d, 0.0]), np.array([0.18 * d, 0.22 * d, -0.16 * d]), 0.018 * d),
        _cylinder_mesh_body(np.array([-0.16 * d, 0.14 * d, 0.0]), np.array([-0.18 * d, 0.22 * d, -0.16 * d]), 0.018 * d),
        _cylinder_mesh_body(np.array([0.16 * d, -0.14 * d, 0.0]), np.array([0.18 * d, -0.22 * d, -0.16 * d]), 0.018 * d),
        _cylinder_mesh_body(np.array([-0.16 * d, -0.14 * d, 0.0]), np.array([-0.18 * d, -0.22 * d, -0.16 * d]), 0.018 * d),
    ]
    mesh_landing_gear = _transform_mesh(_combine_meshes(skid_port + skid_stbd + struts), c_pos, rot_mat)

    # 7. Forward Camera & Sensor Gimbal Assembly
    camera_parts = [
        _cylinder_mesh_body(np.array([0.32 * d, 0.0, 0.0]), np.array([0.40 * d, 0.0, -0.03 * d]), 0.025 * d),
        _sphere_mesh_body(np.array([0.42 * d, 0.0, -0.04 * d]), 0.06 * d),
        _cylinder_mesh_body(np.array([0.42 * d, 0.0, -0.04 * d]), np.array([0.48 * d, 0.0, -0.04 * d]), 0.035 * d),
    ]
    mesh_camera = _transform_mesh(_combine_meshes(camera_parts), c_pos, rot_mat)

    # 8. Forward Heading Indicator Chevron (+X_B)
    mesh_heading = _transform_mesh(_chevron_mesh_body(d), c_pos, rot_mat)

    return {
        "center": c_pos,
        "rotors": world_rotors,
        "nose": nose_world,
        "fuselage": fuselage_pts_w,
        "propellers": propellers_w,
        "front_arms": (world_rotors[0], c_pos, world_rotors[1]),
        "rear_arms": (world_rotors[2], c_pos, world_rotors[3]),
        "mesh_fuselage": mesh_fuselage,
        "mesh_arms": mesh_arms,
        "mesh_front_arms": mesh_front_arms,
        "mesh_rear_arms": mesh_rear_arms,
        "mesh_motors_front": mesh_motors_front,
        "mesh_motors_rear": mesh_motors_rear,
        "mesh_rotor_hubs": mesh_rotor_hubs,
        "mesh_rotor_assemblies": mesh_rotor_assemblies,
        "mesh_props_front": mesh_props_front,
        "mesh_props_rear": mesh_props_rear,
        "mesh_landing_gear": mesh_landing_gear,
        "mesh_camera": mesh_camera,
        "mesh_heading_indicator": mesh_heading,
        "rot_mat": rot_mat,
        "body_rotors": body_rotors,
    }


def _generate_sphere_surface(
    center: np.ndarray | Sequence[float],
    radius: float,
    n_points: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate X, Y, Z coordinate matrices for a 3D sphere surface."""
    u = np.linspace(0.0, 2.0 * np.pi, n_points)
    v = np.linspace(0.0, np.pi, n_points)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def _generate_box_wireframe(
    bounds: Tuple[float, float, float],
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Generate line coordinates for the 12 edges of the 3D arena boundary box."""
    bx, by, bz = bounds
    c = [
        (0.0, 0.0, 0.0),
        (bx, 0.0, 0.0),
        (bx, by, 0.0),
        (0.0, by, 0.0),
        (0.0, 0.0, bz),
        (bx, 0.0, bz),
        (bx, by, bz),
        (0.0, by, bz),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []

    for start_idx, end_idx in edges:
        xs.extend([c[start_idx][0], c[end_idx][0], None])
        ys.extend([c[start_idx][1], c[end_idx][1], None])
        zs.extend([c[start_idx][2], c[end_idx][2], None])

    return xs, ys, zs


def build_arena_3d_figure(
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
    obstacles: Optional[Sequence[ObstacleSphere3D]] = None,
    target: Optional[np.ndarray | Sequence[float]] = None,
    start_pos: Optional[np.ndarray | Sequence[float]] = None,
    trajectory: Optional[Sequence[np.ndarray | Sequence[float]]] = None,
    current_pos: Optional[np.ndarray | Sequence[float]] = None,
    current_vel: Optional[np.ndarray | Sequence[float]] = None,
    attitude_quaternion: Optional[Sequence[float] | np.ndarray] = None,
    wind_vector: Optional[Sequence[float]] = None,
    lidar_rays: Optional[np.ndarray] = None,
    lidar_ranges: Optional[Sequence[float] | np.ndarray] = None,
    show_lidar: bool = True,
    title: str = "3D Drone Flight Arena",
    arm_span: float = 1.8,
    rotor_radius: float = 0.42,
) -> go.Figure:
    """Build an interactive 3D Plotly flight arena visualizer.

    Visualizes:
    - Bounded flight volume with wireframe arena boundary walls
    - Procedural 3D spherical obstacles with proximity clearance zones
    - Goal target waypoint with altitude beacon column and destination label
    - Launch start origin position
    - Real-time or historical drone trajectory
    - Detailed 3D quadrotor airframe model with meshes, propellers, and landing gear
    - Current drone 3D position, heading chevron, and velocity vector
    - 16-ray spherical LiDAR sensor beams and contact points
    """
    fig = go.Figure()

    # 1. Arena Bounding Box Wireframe
    bx, by, bz = bounds
    box_x, box_y, box_z = _generate_box_wireframe(bounds)
    fig.add_trace(
        go.Scatter3d(
            x=box_x,
            y=box_y,
            z=box_z,
            mode="lines",
            line=dict(color="#334155", width=2, dash="dot"),
            name="Arena Boundary",
            hoverinfo="skip",
        )
    )

    # 2. Obstacles (3D Spheres with Proximity / Safety Zones)
    if obstacles:
        for idx, obs in enumerate(obstacles):
            # Physical Obstacle Core
            ox, oy, oz = _generate_sphere_surface(obs.center, obs.radius, n_points=14)
            red_scale = [[0.0, "#991b1b"], [1.0, "#dc2626"]]
            fig.add_trace(
                go.Surface(
                    x=ox,
                    y=oy,
                    z=oz,
                    colorscale=red_scale,
                    showscale=False,
                    opacity=0.60,
                    name=f"Obstacle {idx + 1}",
                    hoverinfo="text",
                    text=(
                        f"Obstacle {idx + 1}<br>"
                        f"Center: [{obs.center[0]:.1f}, {obs.center[1]:.1f}, {obs.center[2]:.1f}]<br>"
                        f"Radius: {obs.radius:.1f} m"
                    ),
                )
            )

            # Obstacle Proximity / Safety Clearance Perimeter Ring
            safety_r = obs.radius + 0.8
            th = np.linspace(0.0, 2.0 * math.pi, 24)
            ring_x = obs.center[0] + safety_r * np.cos(th)
            ring_y = obs.center[1] + safety_r * np.sin(th)
            ring_z = np.full_like(th, obs.center[2])
            fig.add_trace(
                go.Scatter3d(
                    x=ring_x,
                    y=ring_y,
                    z=ring_z,
                    mode="lines",
                    line=dict(color="rgba(239, 68, 68, 0.4)", width=2, dash="dash"),
                    name=f"Safety Margin {idx + 1}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # 3. Target Waypoint with Altitude Beacon Column
    if target is not None:
        # Vertical Altitude Beacon Line from ground to waypoint
        fig.add_trace(
            go.Scatter3d(
                x=[target[0], target[0]],
                y=[target[1], target[1]],
                z=[0.0, target[2] + 1.5],
                mode="lines",
                line=dict(color="#10b981", width=2, dash="dot"),
                name="Waypoint Beacon Column",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Target Waypoint Marker
        fig.add_trace(
            go.Scatter3d(
                x=[target[0]],
                y=[target[1]],
                z=[target[2]],
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color="#10b981",
                    line=dict(color="#ffffff", width=2),
                ),
                text=["WAYPOINT GOAL"],
                textposition="top center",
                textfont=dict(color="#10b981", size=13),
                name="Target Waypoint",
                hoverinfo="text",
                hovertext=f"Target Goal<br>Pos: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]",
            )
        )

    # 4. Start Position
    if start_pos is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[start_pos[0]],
                y=[start_pos[1]],
                z=[start_pos[2]],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=9,
                    color="#06b6d4",
                    line=dict(color="#ffffff", width=1.5),
                ),
                text=["LAUNCH"],
                textposition="bottom center",
                textfont=dict(color="#06b6d4", size=11),
                name="Start Position",
                hoverinfo="text",
                hovertext=f"Launch Origin<br>Pos: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]",
            )
        )

    # 5. Trajectory Line
    if trajectory and len(trajectory) > 1:
        traj_arr = np.asarray(trajectory, dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=traj_arr[:, 0],
                y=traj_arr[:, 1],
                z=traj_arr[:, 2],
                mode="lines",
                line=dict(color="#f59e0b", width=5),
                name="Flight Path",
                hoverinfo="text",
                hovertext=[
                    f"Step {i}<br>Pos: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]"
                    for i, p in enumerate(traj_arr)
                ],
            )
        )

    # 6. LiDAR Beams
    if (
        show_lidar
        and current_pos is not None
        and lidar_rays is not None
        and lidar_ranges is not None
    ):
        c_pos = np.asarray(current_pos, dtype=np.float64)
        lx: List[Optional[float]] = []
        ly: List[Optional[float]] = []
        lz: List[Optional[float]] = []

        for ray_dir, dist in zip(lidar_rays, lidar_ranges):
            end_pt = c_pos + ray_dir * dist
            lx.extend([c_pos[0], end_pt[0], None])
            ly.extend([c_pos[1], end_pt[1], None])
            lz.extend([c_pos[2], end_pt[2], None])

        fig.add_trace(
            go.Scatter3d(
                x=lx,
                y=ly,
                z=lz,
                mode="lines",
                line=dict(color="rgba(0, 242, 254, 0.40)", width=2),
                name="16-Ray LiDAR",
                hoverinfo="skip",
            )
        )

    # 7. Current Quadrotor Airframe: 3D Meshes, Attitude & Telemetry
    if current_pos is not None:
        geom = compute_quadrotor_3d_geometry(
            current_pos,
            quaternion=attitude_quaternion,
            arm_span=arm_span,
            rotor_radius=rotor_radius,
        )

        def _add_mesh(mesh_dict: Dict[str, np.ndarray], color: str, name: str, opacity: float = 1.0, showlegend: bool = False) -> None:
            if len(mesh_dict["x"]) > 0:
                fig.add_trace(
                    go.Mesh3d(
                        x=mesh_dict["x"],
                        y=mesh_dict["y"],
                        z=mesh_dict["z"],
                        i=mesh_dict["i"],
                        j=mesh_dict["j"],
                        k=mesh_dict["k"],
                        color=color,
                        opacity=opacity,
                        name=name,
                        flatshading=True,
                        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.3, specular=0.4),
                        hoverinfo="name",
                        showlegend=showlegend,
                    )
                )

        # 3D Fuselage Pod
        _add_mesh(geom["mesh_fuselage"], "#1e293b", "Fuselage", opacity=0.95, showlegend=True)

        # 3D Carbon Fiber Arms (Front and Rear)
        _add_mesh(geom["mesh_front_arms"], "#ef4444", "Front Arms", opacity=1.0, showlegend=True)
        _add_mesh(geom["mesh_rear_arms"], "#3b82f6", "Rear Arms", opacity=1.0, showlegend=True)

        # Rotor Assemblies (Motors and Hubs)
        _add_mesh(geom["mesh_rotor_assemblies"], "#94a3b8", "Rotor Assemblies", opacity=1.0, showlegend=True)

        # Front & Rear 3D Propeller Blades
        _add_mesh(geom["mesh_props_front"], "#00e5ff", "Front Propellers", opacity=0.9, showlegend=True)
        _add_mesh(geom["mesh_props_rear"], "#3b82f6", "Rear Propellers", opacity=0.9, showlegend=True)

        # Dual Landing Gear Skids & Struts
        _add_mesh(geom["mesh_landing_gear"], "#475569", "Landing Gear", opacity=1.0, showlegend=True)

        # Forward Optical Sensor Gimbal
        _add_mesh(geom["mesh_camera"], "#06b6d4", "Sensor Gimbal & Lens", opacity=1.0, showlegend=True)

        # Forward Heading Indicator Chevron (+X_B)
        _add_mesh(geom["mesh_heading_indicator"], "#fbbf24", "Forward Heading (+X_B)", opacity=1.0, showlegend=True)

        # Center Marker (Drone Trace)
        fig.add_trace(
            go.Scatter3d(
                x=[current_pos[0]],
                y=[current_pos[1]],
                z=[current_pos[2]],
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=8,
                    color="#ed8936",
                    line=dict(color="#ffffff", width=2),
                ),
                text=["DRONE"],
                textposition="bottom center",
                textfont=dict(color="#ed8936", size=12),
                name="Drone",
                hoverinfo="text",
                hovertext=f"Drone Center<br>Pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]",
            )
        )

        # Velocity Vector Arrow
        if current_vel is not None and float(np.linalg.norm(current_vel)) > 0.1:
            vel_endpoint = [
                current_pos[0] + current_vel[0] * 0.8,
                current_pos[1] + current_vel[1] * 0.8,
                current_pos[2] + current_vel[2] * 0.8,
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=[current_pos[0], vel_endpoint[0]],
                    y=[current_pos[1], vel_endpoint[1]],
                    z=[current_pos[2], vel_endpoint[2]],
                    mode="lines",
                    line=dict(color="#f97316", width=4),
                    name="Velocity Vector",
                    hoverinfo="skip",
                )
            )

        # 6-DOF Rigid-body Attitude Coordinate Frame (Roll/Pitch/Yaw)
        if attitude_quaternion is not None:
            rot_mat = quaternion_to_rotation_matrix(attitude_quaternion)
            axis_len = 1.3
            axes_colors = [
                ("#ef4444", "Body X (Roll)"),
                ("#22c55e", "Body Y (Pitch)"),
                ("#3b82f6", "Body Z (Yaw)"),
            ]
            for idx, (col, ax_name) in enumerate(axes_colors):
                vec = rot_mat[:, idx] * axis_len
                fig.add_trace(
                    go.Scatter3d(
                        x=[current_pos[0], current_pos[0] + vec[0]],
                        y=[current_pos[1], current_pos[1] + vec[1]],
                        z=[current_pos[2], current_pos[2] + vec[2]],
                        mode="lines",
                        line=dict(color=col, width=4),
                        name=ax_name,
                        hoverinfo="name",
                        showlegend=False,
                    )
                )

        # Environmental Wind Vector
        if wind_vector is not None and float(np.linalg.norm(wind_vector)) > 0.05:
            wx, wy, wz = [float(v) for v in wind_vector]
            w_origin = [bx * 0.1, by * 0.1, bz * 0.85]
            w_scale = 0.5
            fig.add_trace(
                go.Scatter3d(
                    x=[w_origin[0], w_origin[0] + wx * w_scale],
                    y=[w_origin[1], w_origin[1] + wy * w_scale],
                    z=[w_origin[2], w_origin[2] + wz * w_scale],
                    mode="lines+markers",
                    marker=dict(size=[0, 4], color="#38bdf8"),
                    line=dict(color="#38bdf8", width=5),
                    name=f"Ambient Wind ({np.linalg.norm(wind_vector):.1f} m/s)",
                    hoverinfo="name",
                )
            )

    # Configure 3D Camera, Lighting, and Aerospace Mission-Control Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        title=dict(text=title, font=dict(size=17, color="#f1f5f9"), x=0.02, y=0.98),
        scene=dict(
            xaxis=dict(
                title="X Position (m)",
                range=[0, bx],
                backgroundcolor="#0b0f19",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            yaxis=dict(
                title="Y Position (m)",
                range=[0, by],
                backgroundcolor="#0b0f19",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            zaxis=dict(
                title="Z Altitude (m)",
                range=[0, bz],
                backgroundcolor="#0b0f19",
                gridcolor="#1e293b",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            aspectratio=dict(x=1.0, y=1.0, z=bz / max(bx, by)),
            camera=dict(
                eye=dict(x=1.35, y=-1.35, z=0.85),
                center=dict(x=0.0, y=0.0, z=-0.08),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(11, 15, 25, 0.85)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=11),
        ),
    )

    return fig


def get_available_models(models_dir: str | Path = "artifacts/models") -> List[Path]:
    """Scan the artifacts directory for saved PPO model checkpoints."""
    path = Path(models_dir)
    if not path.exists():
        return []
    return sorted(list(path.glob("*.zip")), key=lambda p: p.stat().st_mtime, reverse=True)


def run_drone_simulation_episode(
    env: Any,
    model: Optional[Any] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single deterministic simulation episode and capture full trajectory."""
    obs, info = env.reset(seed=seed)
    trajectory: List[np.ndarray] = [info["position"].copy()]
    steps_data: List[Dict[str, Any]] = []

    total_reward = 0.0
    terminated = False
    truncated = False
    step_count = 0

    num_rays = int(getattr(env, "num_lidar_rays", 16))
    lidar_range = float(getattr(env, "lidar_range", 20.0))
    act_shape = env.action_space.shape if hasattr(env, "action_space") else (3,)

    raw_lidar_init = obs[-num_rays:] if len(obs) >= num_rays else np.zeros(num_rays)
    lidar_ranges_init = (raw_lidar_init * lidar_range).copy()

    steps_data.append(
        {
            "step": 0,
            "position": info["position"].copy(),
            "velocity": info["velocity"].copy(),
            "speed": float(info.get("speed", 0.0)),
            "angular_velocity": (
                info["angular_velocity"].copy() if "angular_velocity" in info else None
            ),
            "angular_speed": float(info.get("angular_speed", 0.0)),
            "altitude": float(info.get("altitude", info["position"][2])),
            "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
            "min_obstacle_distance": float(info.get("min_obstacle_distance", 0.0)),
            "reward": 0.0,
            "cumulative_reward": 0.0,
            "lidar_ranges": lidar_ranges_init,
            "action": np.zeros(act_shape, dtype=np.float32),
            "collision": False,
            "collision_type": "none",
            "is_success": False,
            "quaternion": info.get("quaternion").copy() if "quaternion" in info else None,
            "wind_vector": info.get("wind_vector"),
            "reward_components": info.get("reward_components"),
        }
    )

    while not (terminated or truncated) and step_count < env.max_steps:
        step_count += 1
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        pos = info["position"].copy()
        vel = info["velocity"].copy()
        trajectory.append(pos)

        raw_lidar_norm = obs[-num_rays:] if len(obs) >= num_rays else np.zeros(num_rays)
        lidar_ranges = (raw_lidar_norm * lidar_range).copy()

        steps_data.append(
            {
                "step": step_count,
                "position": pos,
                "velocity": vel,
                "speed": float(info.get("speed", 0.0)),
                "angular_velocity": (
                    info["angular_velocity"].copy() if "angular_velocity" in info else None
                ),
                "angular_speed": float(info.get("angular_speed", 0.0)),
                "altitude": float(info.get("altitude", pos[2])),
                "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
                "min_obstacle_distance": float(info.get("min_obstacle_distance", 0.0)),
                "reward": float(reward),
                "cumulative_reward": float(total_reward),
                "lidar_ranges": lidar_ranges,
                "action": np.asarray(action, dtype=np.float32).copy(),
                "collision": bool(info.get("collision", False)),
                "collision_type": str(info.get("collision_type", "none")),
                "is_success": bool(info.get("is_success", False) or info.get("success", False)),
                "quaternion": info.get("quaternion").copy() if "quaternion" in info else None,
                "wind_vector": info.get("wind_vector"),
                "reward_components": info.get("reward_components"),
            }
        )

    outcome = "SUCCESS" if steps_data[-1]["is_success"] else "COLLISION / TIMEOUT"

    return {
        "trajectory": trajectory,
        "steps": steps_data,
        "total_reward": total_reward,
        "total_steps": step_count,
        "outcome": outcome,
        "seed": seed,
        "obstacles": getattr(env, "_obstacles", getattr(env, "obstacles", [])),
        "target": getattr(
            env, "target", getattr(env, "_goal", getattr(env, "goal", np.zeros(3)))
        ).copy(),
        "bounds": env.bounds,
        "start_pos": getattr(env, "default_start", getattr(env, "_start_pos", np.zeros(3))).copy(),
        "lidar_rays": env.lidar_rays,
        "wind_vector": info.get("wind_vector"),
        "quaternion": info.get("quaternion"),
    }


def build_training_curve_figure(
    episode_rewards: Sequence[float],
    window_size: int = 10,
    title: str = "PPO Training Progress",
) -> go.Figure:
    """Construct an interactive Plotly training curve with rolling average."""
    fig = go.Figure()
    if not episode_rewards:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#080c14",
            plot_bgcolor="#080c14",
            title=title,
            annotations=[
                dict(
                    text="No training episodes recorded yet.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )
            ],
        )
        return fig

    episodes = list(range(1, len(episode_rewards) + 1))
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=list(episode_rewards),
            mode="lines",
            line=dict(color="rgba(100, 180, 255, 0.4)", width=1.5),
            name="Episode Return",
        )
    )

    if len(episode_rewards) >= window_size:
        rolling = np.convolve(
            np.asarray(episode_rewards, dtype=float),
            np.ones(window_size) / window_size,
            mode="valid",
        )
        rolling_x = list(range(window_size, len(episode_rewards) + 1))
        fig.add_trace(
            go.Scatter(
                x=rolling_x,
                y=rolling.tolist(),
                mode="lines",
                line=dict(color="#00E5FF", width=3),
                name=f"Moving Avg (w={window_size})",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        title=title,
        xaxis_title="Episode",
        yaxis_title="Return (Cumulative Reward)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_comparison_bar_chart(
    metrics_map: Dict[str, Any],
    title: str = "Policy Performance Comparison",
) -> go.Figure:
    """Build side-by-side bar chart comparing PPO vs Random Policy."""
    fig = go.Figure()
    policies = list(metrics_map.keys())
    if not policies:
        return fig

    categories = ["Success Rate (%)", "Collision Rate (%)"]
    colors = ["#00E676", "#FF5252", "#FFD700", "#7C4DFF"]

    for i, policy_name in enumerate(policies):
        m = metrics_map[policy_name]
        succ = (m.success_rate if hasattr(m, "success_rate") else m.get("success_rate", 0.0)) or 0.0
        coll = (
            m.collision_rate if hasattr(m, "collision_rate") else m.get("collision_rate", 0.0)
        ) or 0.0

    for i, policy_name in enumerate(policies):
        m = metrics_map[policy_name]
        succ = (m.success_rate if hasattr(m, "success_rate") else m.get("success_rate", 0.0)) or 0.0
        coll = (
            m.collision_rate if hasattr(m, "collision_rate") else m.get("collision_rate", 0.0)
        ) or 0.0

        fig.add_trace(
            go.Bar(
                name=policy_name,
                x=categories,
                y=[succ * 100.0, coll * 100.0],
                marker_color=colors[i % len(colors)],
                text=[f"{succ * 100:.1f}%", f"{coll * 100:.1f}%"],
                textposition="auto",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        barmode="group",
        title=title,
        yaxis=dict(title="Percentage (%)", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_density_experiment_figure(
    density_results: List[Dict[str, Any]],
    title: str = "Obstacle Density vs Navigation Performance",
) -> go.Figure:
    """Build performance visualization for the obstacle-density experiment."""
    fig = go.Figure()
    if not density_results:
        return fig

    x_labels = [f"{d['obstacle_count']} Obstacles" for d in density_results]
    success_rates = [(d.get("success_rate", 0.0) or 0.0) * 100.0 for d in density_results]
    collision_rates = [(d.get("collision_rate", 0.0) or 0.0) * 100.0 for d in density_results]

    fig.add_trace(
        go.Bar(
            name="Success Rate (%)",
            x=x_labels,
            y=success_rates,
            marker_color="#00E676",
            text=[f"{s:.1f}%" for s in success_rates],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Collision Rate (%)",
            x=x_labels,
            y=collision_rates,
            marker_color="#FF5252",
            text=[f"{c:.1f}%" for c in collision_rates],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        barmode="group",
        title=title,
        yaxis=dict(title="Rate (%)", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_distribution_shift_trajectory_figure(
    fixed_trajectory: Sequence[Sequence[float] | np.ndarray],
    adaptive_trajectory: Sequence[Sequence[float] | np.ndarray],
    bounds: Tuple[float, float, float] = (50.0, 50.0, 25.0),
    obstacles: Optional[Sequence[ObstacleSphere3D]] = None,
    start_pos: Optional[Sequence[float] | np.ndarray] = None,
    target: Optional[Sequence[float] | np.ndarray] = None,
    wind_vector: Optional[Sequence[float]] = None,
    title: str = "Post-Shift Flight Comparison: Fixed Policy vs Adaptive Policy",
) -> go.Figure:
    """Build comparative 3D arena figure showing Fixed vs Adaptive trajectories under shift."""
    fig = build_arena_3d_figure(
        bounds=bounds,
        obstacles=obstacles,
        target=target,
        start_pos=start_pos,
        wind_vector=wind_vector,
        show_lidar=False,
        title=title,
    )

    # Fixed Policy Trajectory (Orange)
    if fixed_trajectory and len(fixed_trajectory) > 1:
        fx = [float(p[0]) for p in fixed_trajectory]
        fy = [float(p[1]) for p in fixed_trajectory]
        fz = [float(p[2]) for p in fixed_trajectory]
        fig.add_trace(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="lines+markers",
                marker=dict(size=2, color="#f97316"),
                line=dict(color="#f97316", width=5, dash="dash"),
                name="Fixed Policy (No Adaptation)",
            )
        )

    # Adaptive Policy Trajectory (Cyan)
    if adaptive_trajectory and len(adaptive_trajectory) > 1:
        ax = [float(p[0]) for p in adaptive_trajectory]
        ay = [float(p[1]) for p in adaptive_trajectory]
        az = [float(p[2]) for p in adaptive_trajectory]
        fig.add_trace(
            go.Scatter3d(
                x=ax,
                y=ay,
                z=az,
                mode="lines+markers",
                marker=dict(size=3, color="#06b6d4"),
                line=dict(color="#06b6d4", width=5),
                name="Adaptive Policy (Online PPO)",
            )
        )

    return fig


def build_recovery_curve_figure(
    episodes: Sequence[int],
    fixed_returns: Sequence[float],
    adaptive_returns: Sequence[float],
    p_pre: float,
    p0: float,
    fixed_t_h: Optional[int] = None,
    adaptive_t_h: Optional[int] = None,
    title: str = "Online Adaptation Recovery Dynamics (Protocol v2.0)",
) -> go.Figure:
    """Build recovery trajectory figure comparing Fixed vs Adaptive returns across post-shift episodes."""
    fig = go.Figure()

    eps = list(episodes)
    degradation = p_pre - p0
    threshold_val = p0 + 0.9 * degradation if degradation > 0 else p_pre

    # Pre-shift baseline P_pre
    fig.add_trace(
        go.Scatter(
            x=[min(eps), max(eps)],
            y=[p_pre, p_pre],
            mode="lines",
            line=dict(color="#10b981", width=2, dash="dash"),
            name=f"Nominal Baseline P_pre ({p_pre:.1f})",
        )
    )

    # Initial Shock baseline P0
    fig.add_trace(
        go.Scatter(
            x=[min(eps), max(eps)],
            y=[p0, p0],
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dot"),
            name=f"Shock Baseline P0 ({p0:.1f})",
        )
    )

    # 90% Recovery Threshold
    if degradation > 0:
        fig.add_trace(
            go.Scatter(
                x=[min(eps), max(eps)],
                y=[threshold_val, threshold_val],
                mode="lines",
                line=dict(color="#f59e0b", width=2, dash="dashdot"),
                name=f"90% Recovery Level ({threshold_val:.1f})",
            )
        )

    # Fixed Policy Returns
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=list(fixed_returns),
            mode="lines+markers",
            marker=dict(size=7, color="#f97316"),
            line=dict(color="#f97316", width=3, dash="dash"),
            name="Fixed Arm (Frozen)",
        )
    )

    # Adaptive Policy Returns
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=list(adaptive_returns),
            mode="lines+markers",
            marker=dict(size=8, color="#06b6d4"),
            line=dict(color="#06b6d4", width=4),
            name="Adaptive Arm (Online PPO)",
        )
    )

    # Highlight shock window (episodes 1-5)
    fig.add_vrect(
        x0=0.5,
        x1=5.5,
        fillcolor="#374151",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="Shared Shock Window (Ep 1-5)",
        annotation_position="top left",
    )

    # Annotate recovery points if available
    if adaptive_t_h is not None and adaptive_t_h < 15:
        idx = adaptive_t_h - 1
        if 0 <= idx < len(adaptive_returns):
            fig.add_annotation(
                x=adaptive_t_h,
                y=adaptive_returns[idx],
                text=f"Adaptive Recovered (T_H={adaptive_t_h})",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#06b6d4",
                font=dict(color="#06b6d4"),
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        title=title,
        xaxis=dict(title="Post-Shift Episode Index (k)", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="Episodic Return"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


__all__ = [
    "build_arena_3d_figure",
    "build_comparison_bar_chart",
    "build_density_experiment_figure",
    "build_distribution_shift_trajectory_figure",
    "build_recovery_curve_figure",
    "build_training_curve_figure",
    "compute_quadrotor_3d_geometry",
    "get_available_models",
    "inspect_model_checkpoint",
    "run_drone_simulation_episode",
]
