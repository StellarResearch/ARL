/**
 * ARL - Adaptive Reinforcement Learning
 * Autonomous Research Flight Deck — Interactive 3D Drone Flight Arena Simulation (Three.js)
 *
 * SCIENTIFIC DISTINCTION & DISCLOSURE:
 * ------------------------------------
 * This client-side browser demonstration implements an illustrative 3-DOF translational
 * point-mass kinematic simulation with linear aerodynamic drag and an analytical 16-ray
 * spherical LiDAR model inside the 30m x 30m x 15m arena.
 *
 * Real-time navigation is driven by a deterministic guidance law (attractive target
 * potential field + repulsive obstacle and boundary fields). It does NOT execute the
 * trained PyTorch neural network policy weights in-browser. To evaluate the genuine
 * trained PPO/SAC policies, launch the Streamlit flight deck via: `adaptive-rl gui`.
 */

(function () {
  'use strict';

  // Physical constants matching src/adaptive_rl/environments/drone.py
  const ARENA_X = 30.0;
  const ARENA_Y = 30.0;
  const ARENA_Z = 15.0;
  const DT = 0.1;
  const MAX_SPEED = 8.0;
  const MAX_ACCEL = 4.0;
  const DRAG_COEFF = 0.05;
  const TARGET_RADIUS = 1.5;
  const COLLISION_RADIUS = 0.8;
  const LIDAR_MAX_RANGE = 20.0;
  const MAX_STEPS = 200;

  // Reward parameters matching Python environment
  const PROGRESS_WEIGHT = 2.0;
  const STEP_PENALTY = -0.05;
  const GOAL_REWARD = 100.0;
  const COLLISION_REWARD = -100.0;
  const ACTION_PENALTY_WEIGHT = 0.01;

  // 16 spherical unit vectors for 3D LiDAR (Fibonacci sphere distribution)
  const NUM_RAYS = 16;
  const LIDAR_RAYS = [];
  for (let i = 0; i < NUM_RAYS; i++) {
    const phi = Math.acos(1 - 2 * (i + 0.5) / NUM_RAYS);
    const theta = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
    LIDAR_RAYS.push(new THREE.Vector3(
      Math.sin(phi) * Math.cos(theta),
      Math.sin(phi) * Math.sin(theta),
      Math.cos(phi)
    ).normalize());
  }

  // Pre-configured obstacles (r = 2.0m)
  const OBSTACLES = [
    { x: 10.0, y: 10.0, z: 5.0, r: 2.0 },
    { x: 18.0, y: 12.0, z: 8.0, r: 2.0 },
    { x: 12.0, y: 20.0, z: 7.0, r: 2.0 },
    { x: 20.0, y: 22.0, z: 9.0, r: 2.0 }
  ];

  // Simulation State
  const simState = {
    isRunning: false,
    step: 0,
    speedMultiplier: 1.0,
    pos: new THREE.Vector3(5.0, 5.0, 3.0),
    vel: new THREE.Vector3(0.0, 0.0, 0.0),
    target: new THREE.Vector3(25.0, 25.0, 10.0),
    cumulativeReward: 0.0,
    lastStepReward: 0.0,
    status: 'READY',
    trajectory: [], // Array of step state snapshots
    viewStepIndex: null // When scrubbing historical playback
  };

  let scene, camera, renderer, container;
  let droneGroup, targetGroup, targetMesh, targetRings = [], obstacleMeshes = [];
  let lidarLineSegments, hitMarkersGroup;
  let trajectoryLine;
  let propellerMeshes = [];
  let animFrameId = null;
  let lastStepTime = 0;
  let prefersReducedMotion = false;

  // Camera Orbit State
  const cameraState = {
    isDragging: false,
    prevMousePos: { x: 0, y: 0 },
    radius: 46.0,
    theta: 0.85,
    phi: 1.15,
    target: new THREE.Vector3(ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2)
  };

  function checkWebGLSupport() {
    try {
      const testCanvas = document.createElement('canvas');
      return !!(window.WebGLRenderingContext && (testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl')));
    } catch (e) {
      return false;
    }
  }

  function renderFallbackUI(message) {
    if (!container) return;
    container.innerHTML = `
      <div class="webgl-fallback-banner" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 340px; padding: 2rem; text-align: center; background: #060a14; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: #94a3b8;">
        <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🛰️</div>
        <h4 style="color: #f1f5f9; margin-bottom: 0.5rem; font-family: monospace;">WEBGL ACCELERATION UNAVAILABLE</h4>
        <p style="max-width: 480px; font-size: 0.88rem; line-height: 1.5; margin-bottom: 1rem;">
          ${message || '3D hardware-accelerated rendering is disabled or unsupported in this browser environment.'}
        </p>
        <div style="font-size: 0.8rem; color: #64748b; font-family: monospace;">
          Standard CPU Python flight telemetry is fully accessible via CLI: <code>adaptive-rl gui</code>
        </div>
      </div>
    `;
  }

  function init() {
    container = document.getElementById('hero-sim-canvas-wrap');
    if (!container) return;

    if (!checkWebGLSupport() || typeof THREE === 'undefined') {
      renderFallbackUI('Three.js or WebGL rendering context could not be loaded.');
      return;
    }

    prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const width = container.clientWidth || 640;
    const height = container.clientHeight || 420;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050811);
    scene.fog = new THREE.FogExp2(0x050811, 0.012);

    // Camera (ENU: +X East, +Y North, +Z Up)
    camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 200);
    camera.up.set(0, 0, 1);
    updateCameraPosition();

    // Renderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(renderer.domElement);
    } catch (err) {
      renderFallbackUI('Failed to create WebGL rendering context.');
      return;
    }

    // Lighting
    buildLighting();

    // Environment Geometry
    buildArena();
    buildObstacles();
    buildTarget();
    buildDetailedDrone();
    buildLidarSystem();
    buildTrajectoryVisualizer();
    buildAxesIndicator();

    // UI & Controls Binding
    bindEvents();
    recordStepSnapshot();
    updateHUD();
    render(0);
  }

  function buildLighting() {
    const ambientLight = new THREE.AmbientLight(0xdbeafe, 0.55);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x00f2fe, 0.85);
    dirLight.position.set(20, 20, 35);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const secondaryLight = new THREE.DirectionalLight(0x3b82f6, 0.45);
    secondaryLight.position.set(-15, -15, 25);
    scene.add(secondaryLight);

    const groundGlow = new THREE.PointLight(0x0284c7, 0.8, 40);
    groundGlow.position.set(ARENA_X / 2, ARENA_Y / 2, 0.5);
    scene.add(groundGlow);
  }

  function buildArena() {
    // Aerospace Flight Deck Grid
    const floorGeom = new THREE.PlaneGeometry(ARENA_X, ARENA_Y);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x070c18,
      roughness: 0.85,
      metalness: 0.2
    });
    const floorMesh = new THREE.Mesh(floorGeom, floorMat);
    floorMesh.position.set(ARENA_X / 2, ARENA_Y / 2, 0);
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    // Primary Grid Lines
    const majorGrid = new THREE.GridHelper(ARENA_X, 30, 0x00f2fe, 0x1e293b);
    majorGrid.rotation.x = Math.PI / 2;
    majorGrid.position.set(ARENA_X / 2, ARENA_Y / 2, 0.02);
    scene.add(majorGrid);

    // Arena Bounding Containment Cage
    const boxGeom = new THREE.BoxGeometry(ARENA_X, ARENA_Y, ARENA_Z);
    const boxEdges = new THREE.EdgesGeometry(boxGeom);
    const boxMat = new THREE.LineBasicMaterial({
      color: 0x334155,
      transparent: true,
      opacity: 0.55,
      linewidth: 1
    });
    const boxWire = new THREE.LineSegments(boxEdges, boxMat);
    boxWire.position.set(ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2);
    scene.add(boxWire);

    // Arena Perimeter Glow Base Line
    const perimeterPoints = [
      new THREE.Vector3(0, 0, 0.05),
      new THREE.Vector3(ARENA_X, 0, 0.05),
      new THREE.Vector3(ARENA_X, ARENA_Y, 0.05),
      new THREE.Vector3(0, ARENA_Y, 0.05),
      new THREE.Vector3(0, 0, 0.05)
    ];
    const perimGeom = new THREE.BufferGeometry().setFromPoints(perimeterPoints);
    const perimMat = new THREE.LineBasicMaterial({ color: 0x0284c7, linewidth: 2 });
    scene.add(new THREE.Line(perimGeom, perimMat));
  }

  function buildObstacles() {
    obstacleMeshes = [];
    OBSTACLES.forEach((obs, idx) => {
      const group = new THREE.Group();
      group.position.set(obs.x, obs.y, obs.z);

      // Core Obstacle Sphere
      const sphereGeom = new THREE.SphereGeometry(obs.r, 28, 28);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: 0x991b1b,
        roughness: 0.35,
        metalness: 0.65,
        transparent: true,
        opacity: 0.88
      });
      const sphereMesh = new THREE.Mesh(sphereGeom, sphereMat);
      sphereMesh.castShadow = true;
      group.add(sphereMesh);

      // Warning Wireframe Cage
      const wireGeom = new THREE.WireframeGeometry(sphereGeom);
      const wireMat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.25 });
      const wireMesh = new THREE.LineSegments(wireGeom, wireMat);
      group.add(wireMesh);

      // Translucent Safety Exclusion Zone (r + COLLISION_RADIUS)
      const safetyGeom = new THREE.SphereGeometry(obs.r + COLLISION_RADIUS, 18, 18);
      const safetyMat = new THREE.MeshBasicMaterial({
        color: 0xf43f5e,
        wireframe: true,
        transparent: true,
        opacity: 0.12
      });
      const safetyMesh = new THREE.Mesh(safetyGeom, safetyMat);
      group.add(safetyMesh);

      // Floor Shadow Projection Circle
      const shadowGeom = new THREE.RingGeometry(obs.r * 0.2, obs.r, 24);
      const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.45, side: THREE.DoubleSide });
      const shadowMesh = new THREE.Mesh(shadowGeom, shadowMat);
      shadowMesh.position.set(0, 0, -obs.z + 0.03);
      group.add(shadowMesh);

      scene.add(group);
      obstacleMeshes.push(group);
    });
  }

  function buildTarget() {
    targetGroup = new THREE.Group();
    targetGroup.position.copy(simState.target);

    // Glowing Target Core
    const geom = new THREE.OctahedronGeometry(TARGET_RADIUS * 0.7, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x10b981,
      emissiveIntensity: 0.75,
      roughness: 0.2,
      metalness: 0.8
    });
    targetMesh = new THREE.Mesh(geom, mat);
    targetGroup.add(targetMesh);

    // Vertical Guidance Beam to Ground
    const beamGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -simState.target.z)
    ]);
    const beamMat = new THREE.LineDashedMaterial({
      color: 0x34d399,
      dashSize: 0.6,
      gapSize: 0.4,
      transparent: true,
      opacity: 0.65
    });
    const beaconLine = new THREE.Line(beamGeom, beamMat);
    beaconLine.computeLineDistances();
    targetGroup.add(beaconLine);

    // Concentric Landing Target Rings on Ground Floor
    targetRings = [];
    [1.0, 1.8, 2.6].forEach((r, idx) => {
      const ringGeom = new THREE.RingGeometry(r - 0.05, r + 0.05, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        transparent: true,
        opacity: 0.55 - idx * 0.12,
        side: THREE.DoubleSide
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.set(0, 0, -simState.target.z + 0.04);
      targetGroup.add(ring);
      targetRings.push(ring);
    });

    scene.add(targetGroup);
  }

  /**
   * Procedural High-Detail Quadrotor Model (6-DOF Coordinate Frame Conventions)
   *
   * Body Frame:
   * +X_B: Forward Heading (Nose)
   * +Y_B: Left
   * +Z_B: Upward (Thrust)
   *
   * Rotor Layout (X-configuration):
   * R1: Front-Right (+d, -d) - CCW (Red accent)
   * R2: Front-Left  (+d, +d) - CW  (Red accent)
   * R3: Rear-Left   (-d, +d) - CCW (Blue accent)
   * R4: Rear-Right  (-d, -d) - CW  (Blue accent)
   */
  function buildDetailedDrone() {
    droneGroup = new THREE.Group();
    propellerMeshes = [];

    const armSpan = 1.6;
    const d = armSpan / (2.0 * Math.sqrt(2.0)); // ~0.5657m

    // Materials
    const carbonMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.3,
      metalness: 0.85
    });
    const metalMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      roughness: 0.2,
      metalness: 0.95
    });
    const frontAccentMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      roughness: 0.3,
      metalness: 0.5
    });
    const rearAccentMat = new THREE.MeshStandardMaterial({
      color: 0x3b82f6,
      roughness: 0.3,
      metalness: 0.5
    });
    const bladeMat = new THREE.MeshStandardMaterial({
      color: 0x00f2fe,
      roughness: 0.15,
      metalness: 0.8,
      transparent: true,
      opacity: 0.85
    });

    // 1. Central Fuselage Assembly
    // Lower Carbon Baseplate
    const lowerPlateGeom = new THREE.BoxGeometry(0.44, 0.32, 0.03);
    const lowerPlate = new THREE.Mesh(lowerPlateGeom, carbonMat);
    lowerPlate.position.set(0, 0, -0.04);
    droneGroup.add(lowerPlate);

    // Upper Carbon Avionics Plate
    const upperPlateGeom = new THREE.BoxGeometry(0.40, 0.28, 0.03);
    const upperPlate = new THREE.Mesh(upperPlateGeom, carbonMat);
    upperPlate.position.set(0, 0, 0.05);
    droneGroup.add(upperPlate);

    // Central Electronics / Battery Block
    const avionicsGeom = new THREE.BoxGeometry(0.32, 0.22, 0.06);
    const avionics = new THREE.Mesh(avionicsGeom, metalMat);
    avionics.position.set(0, 0, 0.005);
    droneGroup.add(avionics);

    // Status Navigation Beacon LED (Green on top)
    const ledGeom = new THREE.SphereGeometry(0.04, 16, 16);
    const ledMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const led = new THREE.Mesh(ledGeom, ledMat);
    led.position.set(0, 0, 0.08);
    droneGroup.add(led);

    // Forward Nose / Gimbal Camera Assembly (+X_B)
    const camGeom = new THREE.CylinderGeometry(0.06, 0.07, 0.12, 16);
    camGeom.rotateZ(Math.PI / 2);
    const camMesh = new THREE.Mesh(camGeom, metalMat);
    camMesh.position.set(0.24, 0, -0.01);
    droneGroup.add(camMesh);

    // Forward Heading Indicator Arrow
    const headingGeom = new THREE.ConeGeometry(0.08, 0.22, 12);
    headingGeom.rotateZ(-Math.PI / 2);
    const headingMesh = new THREE.Mesh(headingGeom, frontAccentMat);
    headingMesh.position.set(0.35, 0, 0.02);
    droneGroup.add(headingMesh);

    // 2. Four Structural Arms (X-Configuration)
    const armGeom = new THREE.CylinderGeometry(0.025, 0.025, armSpan, 12);
    armGeom.rotateZ(Math.PI / 4);
    const arm1 = new THREE.Mesh(armGeom, carbonMat);
    arm1.position.set(0, 0, 0);
    droneGroup.add(arm1);

    const arm2Geom = new THREE.CylinderGeometry(0.025, 0.025, armSpan, 12);
    arm2Geom.rotateZ(-Math.PI / 4);
    const arm2 = new THREE.Mesh(arm2Geom, carbonMat);
    arm2.position.set(0, 0, 0);
    droneGroup.add(arm2);

    // 3. Four Motor Housings, Hubs, and Propellers
    const rotorConfigs = [
      { id: 'R1', x: d, y: -d, isFront: true, isCCW: true },
      { id: 'R2', x: d, y: d, isFront: true, isCCW: false },
      { id: 'R3', x: -d, y: d, isFront: false, isCCW: true },
      { id: 'R4', x: -d, y: -d, isFront: false, isCCW: false }
    ];

    rotorConfigs.forEach(cfg => {
      const motorGroup = new THREE.Group();
      motorGroup.position.set(cfg.x, cfg.y, 0.02);

      // Motor Bell (Cylinder)
      const bellGeom = new THREE.CylinderGeometry(0.07, 0.07, 0.09, 16);
      bellGeom.rotateX(Math.PI / 2);
      const bellMesh = new THREE.Mesh(bellGeom, cfg.isFront ? frontAccentMat : rearAccentMat);
      motorGroup.add(bellMesh);

      // Motor Shaft / Hub
      const shaftGeom = new THREE.CylinderGeometry(0.025, 0.025, 0.04, 12);
      shaftGeom.rotateX(Math.PI / 2);
      const shaftMesh = new THREE.Mesh(shaftGeom, metalMat);
      shaftMesh.position.set(0, 0, 0.06);
      motorGroup.add(shaftMesh);

      // Propeller Hub & 2 Visible Aerodynamic Blades
      const propGroup = new THREE.Group();
      propGroup.position.set(0, 0, 0.08);

      const blade1Geom = new THREE.BoxGeometry(0.36, 0.04, 0.008);
      const blade1 = new THREE.Mesh(blade1Geom, bladeMat);
      propGroup.add(blade1);

      const bladeCenterGeom = new THREE.CylinderGeometry(0.035, 0.035, 0.02, 12);
      bladeCenterGeom.rotateX(Math.PI / 2);
      const bladeCenter = new THREE.Mesh(bladeCenterGeom, metalMat);
      propGroup.add(bladeCenter);

      motorGroup.add(propGroup);
      droneGroup.add(motorGroup);

      propellerMeshes.push({
        group: propGroup,
        isCCW: cfg.isCCW
      });

      // Landing Gear Struts under Motor/Arm Roots
      const strutGeom = new THREE.CylinderGeometry(0.015, 0.015, 0.16, 8);
      strutGeom.rotateX(Math.PI / 2);
      const strut = new THREE.Mesh(strutGeom, carbonMat);
      strut.position.set(cfg.x * 0.75, cfg.y * 0.75, -0.09);
      droneGroup.add(strut);
    });

    droneGroup.position.copy(simState.pos);
    scene.add(droneGroup);
  }

  function buildLidarSystem() {
    // Dynamic 16-ray line segments
    const positions = new Float32Array(NUM_RAYS * 2 * 3);
    const colors = new Float32Array(NUM_RAYS * 2 * 3);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      linewidth: 1.5
    });

    lidarLineSegments = new THREE.LineSegments(geom, mat);
    scene.add(lidarLineSegments);

    // Hit markers group (small point spheres at laser contact)
    hitMarkersGroup = new THREE.Group();
    const markerGeom = new THREE.SphereGeometry(0.12, 8, 8);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0x00f2fe });

    for (let i = 0; i < NUM_RAYS; i++) {
      const marker = new THREE.Mesh(markerGeom, markerMat);
      marker.visible = false;
      hitMarkersGroup.add(marker);
    }
    scene.add(hitMarkersGroup);
  }

  function buildTrajectoryVisualizer() {
    const maxPoints = MAX_STEPS + 20;
    const positions = new Float32Array(maxPoints * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setDrawRange(0, 0);

    const mat = new THREE.LineBasicMaterial({
      color: 0x00f2fe,
      transparent: true,
      opacity: 0.9,
      linewidth: 2.5
    });

    trajectoryLine = new THREE.Line(geom, mat);
    scene.add(trajectoryLine);
  }

  function buildAxesIndicator() {
    // ENU Coordinate Axes Helper (+X Red, +Y Green, +Z Blue)
    const axesHelper = new THREE.AxesHelper(3.0);
    axesHelper.position.set(0.1, 0.1, 0.1);
    scene.add(axesHelper);
  }

  /**
   * Raycast Analytical LiDAR Readings
   * Detects spherical obstacles and arena bounding planes.
   */
  function computeLidar(pos) {
    const readings = [];
    const positions = lidarLineSegments.geometry.attributes.position.array;
    const colors = lidarLineSegments.geometry.attributes.color.array;

    for (let i = 0; i < NUM_RAYS; i++) {
      const rayDir = LIDAR_RAYS[i];
      let minDist = LIDAR_MAX_RANGE;
      let hitType = 'none';

      // Sphere intersections: |pos + t*rayDir - obs.c| = obs.r
      for (let j = 0; j < OBSTACLES.length; j++) {
        const obs = OBSTACLES[j];
        const oc = new THREE.Vector3(pos.x - obs.x, pos.y - obs.y, pos.z - obs.z);
        const b = 2.0 * oc.dot(rayDir);
        const c = oc.lengthSq() - obs.r * obs.r;
        const disc = b * b - 4 * c;
        if (disc >= 0) {
          const t1 = (-b - Math.sqrt(disc)) / 2.0;
          if (t1 > 0 && t1 < minDist) {
            minDist = t1;
            hitType = 'obstacle';
          }
        }
      }

      // Arena boundary limits
      if (rayDir.x > 0 && (ARENA_X - pos.x) / rayDir.x < minDist) {
        minDist = (ARENA_X - pos.x) / rayDir.x;
        hitType = 'boundary';
      } else if (rayDir.x < 0 && -pos.x / rayDir.x < minDist) {
        minDist = -pos.x / rayDir.x;
        hitType = 'boundary';
      }

      if (rayDir.y > 0 && (ARENA_Y - pos.y) / rayDir.y < minDist) {
        minDist = (ARENA_Y - pos.y) / rayDir.y;
        hitType = 'boundary';
      } else if (rayDir.y < 0 && -pos.y / rayDir.y < minDist) {
        minDist = -pos.y / rayDir.y;
        hitType = 'boundary';
      }

      if (rayDir.z > 0 && (ARENA_Z - pos.z) / rayDir.z < minDist) {
        minDist = (ARENA_Z - pos.z) / rayDir.z;
        hitType = 'boundary';
      } else if (rayDir.z < 0 && -pos.z / rayDir.z < minDist) {
        minDist = -pos.z / rayDir.z;
        hitType = 'boundary';
      }

      // Clamp distance
      minDist = Math.max(0.0, Math.min(LIDAR_MAX_RANGE, minDist));
      readings.push(minDist);

      // Line Segment Coordinates
      const idx = i * 6;
      positions[idx] = pos.x;
      positions[idx + 1] = pos.y;
      positions[idx + 2] = pos.z;

      const hitX = pos.x + rayDir.x * minDist;
      const hitY = pos.y + rayDir.y * minDist;
      const hitZ = pos.z + rayDir.z * minDist;

      positions[idx + 3] = hitX;
      positions[idx + 4] = hitY;
      positions[idx + 5] = hitZ;

      // Color coding: Red if dangerously close (<4m), Amber (4-10m), Cyan (>10m)
      const normDist = minDist / LIDAR_MAX_RANGE;
      if (normDist < 0.2) {
        // Red
        colors[idx + 3] = 0.95; colors[idx + 4] = 0.2; colors[idx + 5] = 0.2;
      } else if (normDist < 0.5) {
        // Amber
        colors[idx + 3] = 0.96; colors[idx + 4] = 0.62; colors[idx + 5] = 0.07;
      } else {
        // Cyan
        colors[idx + 3] = 0.0; colors[idx + 4] = 0.95; colors[idx + 5] = 1.0;
      }

      // Hit Marker Sphere Update
      const marker = hitMarkersGroup.children[i];
      if (marker) {
        if (minDist < LIDAR_MAX_RANGE - 0.1) {
          marker.position.set(hitX, hitY, hitZ);
          marker.visible = true;
          marker.material.color.setRGB(colors[idx + 3], colors[idx + 4], colors[idx + 5]);
        } else {
          marker.visible = false;
        }
      }
    }

    lidarLineSegments.geometry.attributes.position.needsUpdate = true;
    lidarLineSegments.geometry.attributes.color.needsUpdate = true;
    return readings;
  }

  /**
   * Main Physics Step Function
   * Strictly aligns with Semi-Implicit Euler integration, post-step distance,
   * post-step goal detection, post-step collision detection, and finite guarantees.
   */
  function stepSim() {
    if (!simState.isRunning) return;

    simState.step += 1;

    // 1. Guidance Law: Target attraction + Obstacle repulsion
    const toGoal = new THREE.Vector3().subVectors(simState.target, simState.pos);
    const distBefore = toGoal.length();

    // Goal Attraction
    const desiredVel = toGoal.clone().normalize().multiplyScalar(Math.min(distBefore, MAX_SPEED));
    const accel = new THREE.Vector3().subVectors(desiredVel, simState.vel).multiplyScalar(2.0);

    // Obstacle Repulsion
    OBSTACLES.forEach(obs => {
      const obsPos = new THREE.Vector3(obs.x, obs.y, obs.z);
      const toObs = new THREE.Vector3().subVectors(simState.pos, obsPos);
      const dist = toObs.length();
      const safeDist = obs.r + 2.5;
      if (dist < safeDist && dist > 1e-4) {
        const repForce = toObs.normalize().multiplyScalar((safeDist - dist) * 14.0);
        accel.add(repForce);
      }
    });

    // Arena Boundary Repulsion
    const boundaryBuffer = 3.0;
    if (simState.pos.x < boundaryBuffer) accel.x += (boundaryBuffer - simState.pos.x) * 6.0;
    if (simState.pos.x > ARENA_X - boundaryBuffer) accel.x -= (simState.pos.x - (ARENA_X - boundaryBuffer)) * 6.0;
    if (simState.pos.y < boundaryBuffer) accel.y += (boundaryBuffer - simState.pos.y) * 6.0;
    if (simState.pos.y > ARENA_Y - boundaryBuffer) accel.y -= (simState.pos.y - (ARENA_Y - boundaryBuffer)) * 6.0;
    if (simState.pos.z < 2.0) accel.z += (2.0 - simState.pos.z) * 8.0;
    if (simState.pos.z > ARENA_Z - 2.0) accel.z -= (simState.pos.z - (ARENA_Z - 2.0)) * 8.0;

    // Clamp acceleration magnitude
    if (accel.length() > MAX_ACCEL) {
      accel.normalize().multiplyScalar(MAX_ACCEL);
    }

    // 2. Numerical Integration: Semi-Implicit Euler with linear aerodynamic drag damping
    // effective_acc = a_cmd - c_d * v
    // v_{t+1} = clamp(v_t + effective_acc * dt, speed <= max_velocity)
    // p_{t+1} = p_t + v_{t+1} * dt
    const effectiveAcc = accel.clone().addScaledVector(simState.vel, -DRAG_COEFF);
    simState.vel.addScaledVector(effectiveAcc, DT);
    if (simState.vel.length() > MAX_SPEED) {
      simState.vel.normalize().multiplyScalar(MAX_SPEED);
    }

    // Step position with post-integration velocity
    simState.pos.addScaledVector(simState.vel, DT);

    // Guarantee finite numerical coordinates
    if (!Number.isFinite(simState.pos.x) || !Number.isFinite(simState.pos.y) || !Number.isFinite(simState.pos.z)) {
      simState.pos.set(5.0, 5.0, 3.0);
      simState.vel.set(0.0, 0.0, 0.0);
    }

    // 3. Post-step Distance to Goal & Progress Calculation
    const distAfter = simState.pos.distanceTo(simState.target);
    const distDelta = distBefore - distAfter; // Positive when moving closer to target

    // 4. Post-step Collision Detection
    let isCollision = false;
    for (let i = 0; i < OBSTACLES.length; i++) {
      const obs = OBSTACLES[i];
      const d = simState.pos.distanceTo(new THREE.Vector3(obs.x, obs.y, obs.z));
      if (d <= obs.r + COLLISION_RADIUS) {
        isCollision = true;
        break;
      }
    }
    if (!isCollision) {
      if (
        simState.pos.x - COLLISION_RADIUS <= 0.0 || simState.pos.x + COLLISION_RADIUS >= ARENA_X ||
        simState.pos.y - COLLISION_RADIUS <= 0.0 || simState.pos.y + COLLISION_RADIUS >= ARENA_Y ||
        simState.pos.z - COLLISION_RADIUS <= 0.0 || simState.pos.z + COLLISION_RADIUS >= ARENA_Z
      ) {
        isCollision = true;
      }
    }

    // 5. Post-step Success (Goal Reached) Check
    const isSuccess = distAfter <= TARGET_RADIUS;

    // 6. Reward & Termination (matching Python DroneNavigation3DEnv)
    let stepReward = 0.0;
    if (isCollision) {
      stepReward = COLLISION_REWARD;
      simState.isRunning = false;
      simState.status = 'COLLISION DETECTED';
    } else if (isSuccess) {
      stepReward = GOAL_REWARD;
      simState.isRunning = false;
      simState.status = 'SUCCESS (GOAL REACHED)';
    } else {
      const progressReward = PROGRESS_WEIGHT * distDelta;
      const effortNormalized = accel.lengthSq() / (MAX_ACCEL * MAX_ACCEL);
      const effortPenalty = ACTION_PENALTY_WEIGHT * effortNormalized;
      stepReward = progressReward + STEP_PENALTY - effortPenalty;

      if (simState.step >= MAX_STEPS) {
        simState.isRunning = false;
        simState.status = 'TRUNCATED (MAX STEPS)';
      } else {
        simState.status = 'NAVIGATING';
      }
    }

    simState.cumulativeReward += stepReward;
    simState.lastStepReward = stepReward;

    // 7. Update Drone Attitude and Mesh Position
    droneGroup.position.copy(simState.pos);

    // Realistic banking attitude according to velocity and acceleration
    const speed = simState.vel.length();
    if (speed > 0.05) {
      droneGroup.rotation.z = Math.atan2(simState.vel.y, simState.vel.x);
      droneGroup.rotation.x = THREE.MathUtils.clamp(-accel.y * 0.07, -0.4, 0.4);
      droneGroup.rotation.y = THREE.MathUtils.clamp(accel.x * 0.07, -0.4, 0.4);
    }

    // 8. Record Trajectory Snapshot
    recordStepSnapshot();
    updateTrajectoryLine();

    // 9. LiDAR Sensor Calculations
    computeLidar(simState.pos);

    // 10. HUD update
    updateHUD();
  }

  function recordStepSnapshot() {
    simState.trajectory.push({
      step: simState.step,
      pos: simState.pos.clone(),
      vel: simState.vel.clone(),
      reward: simState.cumulativeReward,
      distToGoal: simState.pos.distanceTo(simState.target),
      status: simState.status
    });
  }

  function updateTrajectoryLine(countOverride) {
    const positions = trajectoryLine.geometry.attributes.position.array;
    const count = (typeof countOverride === 'number') ? countOverride : simState.trajectory.length;
    for (let i = 0; i < count; i++) {
      const p = simState.trajectory[i].pos;
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
    }
    trajectoryLine.geometry.setDrawRange(0, count);
    trajectoryLine.geometry.attributes.position.needsUpdate = true;
  }

  function updateHUD(displayedSnapshot) {
    const snap = displayedSnapshot || {
      step: simState.step,
      pos: simState.pos,
      vel: simState.vel,
      reward: simState.cumulativeReward,
      distToGoal: simState.pos.distanceTo(simState.target),
      status: simState.status
    };

    const stepEl = document.getElementById('hud-step');
    const posEl = document.getElementById('hud-pos');
    const velEl = document.getElementById('hud-vel');
    const distEl = document.getElementById('hud-dist');
    const rewEl = document.getElementById('hud-reward');
    const statusEl = document.getElementById('sim-status-text');
    const scrubEl = document.getElementById('sim-scrubber');

    if (stepEl) stepEl.textContent = `Step ${snap.step} / ${MAX_STEPS}`;
    if (posEl) posEl.textContent = `(${snap.pos.x.toFixed(1)}, ${snap.pos.y.toFixed(1)}, ${snap.pos.z.toFixed(1)})`;
    if (velEl) velEl.textContent = `${snap.vel.length().toFixed(1)} m/s`;
    if (distEl) distEl.textContent = `${snap.distToGoal.toFixed(1)} m`;
    if (rewEl) rewEl.textContent = snap.reward.toFixed(1);
    if (statusEl) {
      statusEl.textContent = snap.status;
      if (snap.status.includes('SUCCESS')) {
        statusEl.style.color = '#10b981';
      } else if (snap.status.includes('COLLISION')) {
        statusEl.style.color = '#ef4444';
      } else {
        statusEl.style.color = '#00f2fe';
      }
    }
    if (scrubEl) {
      scrubEl.max = Math.max(1, simState.trajectory.length - 1);
      scrubEl.value = snap.step;
      scrubEl.disabled = simState.isRunning;
    }
  }

  function resetSim() {
    simState.isRunning = false;
    simState.step = 0;
    simState.pos.set(5.0, 5.0, 3.0);
    simState.vel.set(0.0, 0.0, 0.0);
    simState.cumulativeReward = 0.0;
    simState.lastStepReward = 0.0;
    simState.status = 'READY';
    simState.trajectory = [];
    simState.viewStepIndex = null;

    droneGroup.position.copy(simState.pos);
    droneGroup.rotation.set(0, 0, 0);

    recordStepSnapshot();
    updateTrajectoryLine();
    computeLidar(simState.pos);
    updateHUD();

    const startBtn = document.getElementById('btn-sim-start');
    if (startBtn) startBtn.textContent = 'START';
  }

  function applyScrubIndex(idx) {
    if (idx < 0 || idx >= simState.trajectory.length) return;
    simState.viewStepIndex = idx;
    const snap = simState.trajectory[idx];
    droneGroup.position.copy(snap.pos);
    updateTrajectoryLine(idx + 1);
    computeLidar(snap.pos);
    updateHUD(snap);
  }

  function updateCameraPosition() {
    const cx = cameraState.target.x + cameraState.radius * Math.sin(cameraState.phi) * Math.cos(cameraState.theta);
    const cy = cameraState.target.y + cameraState.radius * Math.sin(cameraState.phi) * Math.sin(cameraState.theta);
    const cz = cameraState.target.z + cameraState.radius * Math.cos(cameraState.phi);
    camera.position.set(cx, cy, cz);
    camera.lookAt(cameraState.target);
  }

  function bindEvents() {
    const startBtn = document.getElementById('btn-sim-start');
    const pauseBtn = document.getElementById('btn-sim-pause');
    const resetBtn = document.getElementById('btn-sim-reset');
    const scrubEl = document.getElementById('sim-scrubber');
    const camResetBtn = document.getElementById('btn-sim-cam-reset');
    const speedHalfBtn = document.getElementById('btn-sim-speed-half');
    const speed1Btn = document.getElementById('btn-sim-speed-1');
    const speed2Btn = document.getElementById('btn-sim-speed-2');

    if (startBtn) {
      startBtn.addEventListener('click', () => {
        if (simState.status.includes('SUCCESS') || simState.status.includes('COLLISION') || simState.step >= MAX_STEPS) {
          resetSim();
        }
        if (simState.viewStepIndex !== null && simState.viewStepIndex < simState.trajectory.length - 1) {
          // If resuming from scrubbed past, restore to current real head
          simState.viewStepIndex = null;
          updateTrajectoryLine();
        }
        simState.isRunning = true;
        simState.status = 'NAVIGATING';
        startBtn.textContent = 'RESUME';
        updateHUD();
      });
    }

    if (pauseBtn) {
      pauseBtn.addEventListener('click', () => {
        simState.isRunning = false;
        simState.status = 'PAUSED';
        updateHUD();
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        resetSim();
      });
    }

    if (camResetBtn) {
      camResetBtn.addEventListener('click', () => {
        cameraState.radius = 46.0;
        cameraState.theta = 0.85;
        cameraState.phi = 1.15;
        updateCameraPosition();
      });
    }

    // Playback Speed Controls
    function setSpeed(multiplier, activeBtn) {
      simState.speedMultiplier = multiplier;
      [speedHalfBtn, speed1Btn, speed2Btn].forEach(b => {
        if (b) b.classList.remove('active');
      });
      if (activeBtn) activeBtn.classList.add('active');
    }

    if (speedHalfBtn) speedHalfBtn.addEventListener('click', () => setSpeed(0.5, speedHalfBtn));
    if (speed1Btn) speed1Btn.addEventListener('click', () => setSpeed(1.0, speed1Btn));
    if (speed2Btn) speed2Btn.addEventListener('click', () => setSpeed(2.0, speed2Btn));

    // Interactive Scrubber
    if (scrubEl) {
      scrubEl.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        applyScrubIndex(val);
      });
    }

    // Keyboard Access
    window.addEventListener('keydown', (e) => {
      // Avoid intercepting input if typing in an input or textarea
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

      if (e.code === 'Space') {
        e.preventDefault();
        if (simState.isRunning) {
          simState.isRunning = false;
          simState.status = 'PAUSED';
        } else {
          if (simState.status.includes('SUCCESS') || simState.status.includes('COLLISION') || simState.step >= MAX_STEPS) {
            resetSim();
          }
          simState.isRunning = true;
          simState.status = 'NAVIGATING';
        }
        updateHUD();
      } else if (e.key === 'r' || e.key === 'R') {
        resetSim();
      } else if (e.key === 'c' || e.key === 'C') {
        cameraState.radius = 46.0;
        cameraState.theta = 0.85;
        cameraState.phi = 1.15;
        updateCameraPosition();
      }
    });

    // Pointer Drag & Touch Orbit Controls
    renderer.domElement.addEventListener('mousedown', (e) => {
      cameraState.isDragging = true;
      cameraState.prevMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => { cameraState.isDragging = false; });

    window.addEventListener('mousemove', (e) => {
      if (!cameraState.isDragging) return;
      const deltaX = e.clientX - cameraState.prevMousePos.x;
      const deltaY = e.clientY - cameraState.prevMousePos.y;
      cameraState.prevMousePos = { x: e.clientX, y: e.clientY };

      cameraState.theta -= deltaX * 0.007;
      cameraState.phi = Math.max(0.15, Math.min(Math.PI / 2 - 0.02, cameraState.phi - deltaY * 0.007));
      updateCameraPosition();
    });

    // Wheel Zoom
    renderer.domElement.addEventListener('wheel', (e) => {
      e.preventDefault();
      cameraState.radius = THREE.MathUtils.clamp(cameraState.radius + e.deltaY * 0.04, 15.0, 90.0);
      updateCameraPosition();
    }, { passive: false });

    // Touch Support
    let prevTouchDist = 0;
    renderer.domElement.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        cameraState.isDragging = true;
        cameraState.prevMousePos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        prevTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    });

    renderer.domElement.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && cameraState.isDragging) {
        const deltaX = e.touches[0].clientX - cameraState.prevMousePos.x;
        const deltaY = e.touches[0].clientY - cameraState.prevMousePos.y;
        cameraState.prevMousePos = { x: e.touches[0].clientX, y: e.touches[0].clientY };

        cameraState.theta -= deltaX * 0.007;
        cameraState.phi = Math.max(0.15, Math.min(Math.PI / 2 - 0.02, cameraState.phi - deltaY * 0.007));
        updateCameraPosition();
      } else if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        const diff = prevTouchDist - dist;
        cameraState.radius = THREE.MathUtils.clamp(cameraState.radius + diff * 0.05, 15.0, 90.0);
        prevTouchDist = dist;
        updateCameraPosition();
      }
    }, { passive: false });

    renderer.domElement.addEventListener('touchend', () => { cameraState.isDragging = false; });

    // Responsive Canvas Resize
    window.addEventListener('resize', () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
  }

  function render(time) {
    animFrameId = requestAnimationFrame(render);

    // Dynamic animations if reduced motion is not requested
    if (!prefersReducedMotion) {
      // Animate Target Octahedron rotation & pulse
      if (targetMesh) {
        targetMesh.rotation.y += 0.015;
        targetMesh.rotation.x += 0.008;
        const scale = 1.0 + Math.sin(time * 0.004) * 0.12;
        targetMesh.scale.set(scale, scale, scale);
      }

      // Animate target ground landing radar rings
      if (targetRings && targetRings.length > 0) {
        targetRings.forEach((r, idx) => {
          const ringPulse = 1.0 + Math.sin(time * 0.003 + idx * 1.2) * 0.08;
          r.scale.set(ringPulse, ringPulse, 1.0);
        });
      }

      // Spin Propeller Blades when running
      if (simState.isRunning && propellerMeshes.length > 0) {
        const spinSpeed = 0.45 * simState.speedMultiplier;
        propellerMeshes.forEach(p => {
          p.group.rotation.z += p.isCCW ? spinSpeed : -spinSpeed;
        });
      }
    }

    // Step physics at 10Hz (100ms interval scaled by speedMultiplier)
    const stepInterval = 100 / simState.speedMultiplier;
    if (time - lastStepTime > stepInterval) {
      if (simState.isRunning) {
        stepSim();
      }
      lastStepTime = time;
    }

    renderer.render(scene, camera);
  }

  // Export module for testing and inspection
  window.ARLSimulation = {
    getSimState: () => ({ ...simState }),
    stepSim,
    resetSim,
    applyScrubIndex,
    computeLidar,
    constants: {
      ARENA_X, ARENA_Y, ARENA_Z, DT, MAX_SPEED, MAX_ACCEL, DRAG_COEFF,
      TARGET_RADIUS, COLLISION_RADIUS, LIDAR_MAX_RANGE, MAX_STEPS,
      PROGRESS_WEIGHT, STEP_PENALTY, GOAL_REWARD, COLLISION_REWARD
    }
  };

  // Self-initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
