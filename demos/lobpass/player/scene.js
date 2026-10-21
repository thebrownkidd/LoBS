// Renderer, lights, court (procedural maple + markings), hoop, backboard and stanchion. All coordinates are in the
// court frame (Z up); main.js puts everything inside one group rotated into three.js's Y-up world.
import * as THREE from "three";

// deterministic pseudo-random numbers (the look must not depend on Math.random)
export function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

export function createRenderer(canvas, capture) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: capture, powerPreference: "high-performance" });
  r.setPixelRatio(capture ? 1 : Math.min(window.devicePixelRatio || 1, 2));
  r.shadowMap.enabled = true;
  r.shadowMap.type = THREE.PCFSoftShadowMap;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.toneMappingExposure = 1.0;
  r.outputColorSpace = THREE.SRGBColorSpace;
  return r;
}

export function createBackground() {
  const c = document.createElement("canvas");
  c.width = 4; c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, "#1b2130"); grad.addColorStop(1, "#0e1117");
  g.fillStyle = grad; g.fillRect(0, 0, 4, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function createLights(scene) {
  const hemi = new THREE.HemisphereLight(0xdde6ff, 0x3a2e22, 0.6 * Math.PI);
  hemi.position.set(0, 0, 1);  // court-frame up
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(4, -5, 16);
  key.target.position.set(4, 0, 0);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  const sc = key.shadow.camera;
  sc.left = -11; sc.right = 11; sc.top = 11; sc.bottom = -11; sc.near = 1; sc.far = 40;
  key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02; key.shadow.radius = 3;
  scene.add(key, key.target);
  for (const [x, y] of [[1, -5], [1, 5], [9, -5], [9, 5]]) {  // arena spots, no shadows
    const s = new THREE.SpotLight(0xfff4e0, 220, 0, THREE.MathUtils.degToRad(35), 0.6, 2);
    s.position.set(x, y, 12); s.target.position.set(x, y * 0.3, 0);
    scene.add(s, s.target);
  }
}

// ---------------------------------------------------------------- court
const X0 = -3.6, X1 = 14.7, Y0 = -9.6, Y1 = 9.6;  // texture extent in metres (court + apron)

export function createCourt(scene, K, lines) {
  const W = 4096, H = 4096;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  const px = (x) => ((x - X0) / (X1 - X0)) * W;
  const py = (y) => ((Y1 - y) / (Y1 - Y0)) * H;
  g.fillStyle = "#2b2b2b"; g.fillRect(0, 0, W, H);  // apron
  const rnd = lcg(12345);
  const courtX0 = K.x0, courtX1 = K.x1, courtY = K.y;
  // planks run along x, 0.06 m wide, two-tone grain with seams
  const plankH = (0.06 / (Y1 - Y0)) * H;
  for (let y = -courtY; y < courtY - 1e-6; y += 0.06) {
    const shade = 0.88 + rnd() * 0.16;
    const base = [201, 143, 79].map((v) => Math.min(255, Math.round(v * shade)));
    g.fillStyle = `rgb(${base[0]},${base[1]},${base[2]})`;
    g.fillRect(px(courtX0), py(y + 0.06), px(courtX1) - px(courtX0), plankH + 1);
    g.fillStyle = "rgba(90,50,20,0.55)";
    g.fillRect(px(courtX0), py(y), px(courtX1) - px(courtX0), 1);
    let x = courtX0 + rnd() * 1.2;
    while (x < courtX1) {  // plank end seams
      g.fillRect(px(x), py(y + 0.06), 1, plankH);
      x += 1.2 + rnd() * 1.2;
    }
    for (let k = 0; k < 4; k++) {  // grain streaks
      g.strokeStyle = `rgba(110,65,25,${0.06 + rnd() * 0.08})`;
      g.lineWidth = 1;
      const gy = py(y + 0.06) + rnd() * plankH;
      g.beginPath(); g.moveTo(px(courtX0), gy); g.lineTo(px(courtX1), gy + (rnd() - 0.5) * 3); g.stroke();
    }
  }
  // the paint
  g.globalAlpha = 0.93; g.fillStyle = "#9e2a2b";
  g.fillRect(px(courtX0), py(K.lane_y), px(K.ft_x) - px(courtX0), py(-K.lane_y) - py(K.lane_y));
  g.globalAlpha = 1.0;
  // markings
  g.strokeStyle = "#ffffff"; g.lineCap = "butt"; g.lineJoin = "miter";
  g.lineWidth = (K.line_w / (X1 - X0)) * W;
  for (const polys of Object.values(lines)) {
    for (const pl of polys) {
      g.beginPath();
      pl.forEach(([x, y], i) => (i ? g.lineTo(px(x), py(y)) : g.moveTo(px(x), py(y))));
      g.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.generateMipmaps = true;
  const mat = new THREE.MeshPhysicalMaterial({ map: tex, roughness: 0.38, metalness: 0.0, clearcoat: 0.5, clearcoatRoughness: 0.35 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(X1 - X0, Y1 - Y0), mat);
  floor.position.set((X0 + X1) / 2, 0, 0);
  floor.receiveShadow = true;
  scene.add(floor);
  // a dark ground beyond the apron so the horizon is not a void
  const far = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x14171d, roughness: 1 }));
  far.position.set(5, 0, -0.02);
  far.receiveShadow = true;
  scene.add(far);
}

// ---------------------------------------------------------------- hoop
export function createHoop(scene, K) {
  const hoop = new THREE.Group();
  scene.add(hoop);
  // backboard: glass with a white border and shooter's square
  const [bx, by, bz] = K.board;
  const glass = new THREE.Mesh(
    new THREE.BoxGeometry(bx, by, bz),
    new THREE.MeshPhysicalMaterial({ color: 0xcfe8ff, transparent: true, opacity: 0.22, roughness: 0.05, metalness: 0, clearcoat: 1 })
  );
  glass.position.set(...K.board_center);
  hoop.add(glass);
  const lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const t = 0.03;
  const front = K.board_center[0] + bx / 2 + 0.002;
  const frame = (w, h, cy, cz, th) => {
    for (const [sx, sy, sw, sh] of [[0, h / 2, w, th], [0, -h / 2, w, th], [-w / 2, 0, th, h], [w / 2, 0, th, h]]) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(sh, sw), lineMat);  // after the y rotation: local x is vertical
      m.rotation.y = Math.PI / 2;  // plane faces +x
      m.position.set(front, cy + sx, cz + sy);
      hoop.add(m);
    }
  };
  frame(by, bz, K.board_center[1], K.board_center[2], t);
  frame(0.59, 0.45, 0, 3.048 + 0.1, 0.02);  // shooter's square above the rim
  // rim + bracket (the rim group flexes on a slam)
  const rimPivot = new THREE.Group();
  rimPivot.position.set(-0.30, 0, K.rim_z);
  hoop.add(rimPivot);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(K.rim_r, 0.011, 16, 96),
    new THREE.MeshStandardMaterial({ color: 0xd9531e, metalness: 0.6, roughness: 0.35 })
  );
  ring.position.set(0.30, 0, 0);
  ring.castShadow = true;
  rimPivot.add(ring);
  const bracket = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.05, 0.03), new THREE.MeshStandardMaterial({ color: 0xd9531e, metalness: 0.6, roughness: 0.4 }));
  bracket.position.set(0.0, 0, 0);
  bracket.castShadow = true;
  rimPivot.add(bracket);
  // stanchion: padded base and an arm to the backboard
  const dark = new THREE.MeshStandardMaterial({ color: 0x3a3f4a, roughness: 0.7, metalness: 0.2 });
  const pad = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1.4, 2.0), new THREE.MeshStandardMaterial({ color: 0x2b303a, roughness: 0.85 }));
  pad.position.set(-2.4, 0, 1.0);
  pad.castShadow = pad.receiveShadow = true;
  hoop.add(pad);
  const pole = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.3, 3.3), dark);
  pole.position.set(-2.4, 0, 3.0);
  pole.castShadow = true;
  hoop.add(pole);
  const arm = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.3, 0.28), dark);
  arm.position.set(-1.4, 0, 4.5);
  arm.castShadow = true;
  hoop.add(arm);
  return { hoop, rimPivot, ring };
}

// damped rim flex after a slam: front dips 3 degrees, omega 25 rad/s, zeta 0.25 (visual only)
export function rimAngle(t) {
  if (t < 0) return 0;
  const A = THREE.MathUtils.degToRad(3.0), w = 25, z = 0.25;
  return A * Math.exp(-z * w * t) * Math.cos(w * Math.sqrt(1 - z * z) * t);
}
