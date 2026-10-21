// Stylized mannequin built on the skeleton exported by the baker (rig.joints / parent / offset). Meshes follow SPEC B2:
// ellipsoid torso pieces, capsule limbs, box hands and shoes. The baked data only sets joint rotations and the
// pelvis position, so what you see is exactly what the Python QA measured.
import * as THREE from "three";

const SKIN = 0xc8a27e;
const TEAMS = {
  offense: { jersey: 0x1f4fd1, shorts: 0x163a9e, shoe: 0xffffff, sole: 0x1f4fd1, text: "#ffffff" },
  defense: { jersey: 0xf2f2f2, shorts: 0xc9c9c9, shoe: 0xffffff, sole: 0x444444, text: "#222222" },
};

function mat(color, rough = 0.62) {
  return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.0 });
}

function ellipsoid(rx, ry, rz, material) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 18), material);
  m.scale.set(rx, ry, rz);
  return m;
}

// capsule along local -z from the joint, ``len`` long including the caps
function limb(r, len, material) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(len - 2 * r, 0.001), 6, 16), material);
  m.rotation.x = Math.PI / 2;  // CapsuleGeometry runs along Y; lay it along Z
  const g = new THREE.Group();
  g.add(m);
  g.position.z = -len / 2;
  return g;
}

function numberPlane(num, team, H, flip) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = team.text; g.font = "700 92px Inter, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(String(num), 64, 70);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const p = new THREE.Mesh(new THREE.PlaneGeometry(0.11 * H, 0.11 * H), new THREE.MeshBasicMaterial({ map: t, transparent: true }));
  // plane normal -> +x (front) or -x (back), plane up -> joint +z (the skeleton is Z-up)
  const basis = flip
    ? new THREE.Matrix4().makeBasis(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0))
    : new THREE.Matrix4().makeBasis(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0));
  p.quaternion.setFromRotationMatrix(basis);
  return p;
}

export function buildCharacter(def, rig, parentGroup) {
  const H = def.H;
  const team = TEAMS[def.team];
  const jersey = mat(team.jersey, 0.7), shorts = mat(team.shorts, 0.75), skin = mat(SKIN, 0.6);
  const shoe = mat(team.shoe, 0.5), sole = mat(team.sole, 0.8), hair = mat(0x1a1410, 0.9);
  const joints = rig.joints.map(() => new THREE.Group());
  const index = Object.fromEntries(rig.joints.map((n, i) => [n, i]));
  rig.parent.forEach((p, i) => {
    if (p < 0) return;
    joints[p].add(joints[i]);
    joints[i].position.set(rig.offset[i][0] * H, rig.offset[i][1] * H, rig.offset[i][2] * H);
  });
  const add = (joint, mesh, x = 0, y = 0, z = 0) => {
    mesh.position.set(mesh.position.x + x * H, mesh.position.y + y * H, mesh.position.z + z * H);
    mesh.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    joints[index[joint]].add(mesh);
  };
  const [aU, aF] = rig.arm, [lT, lS] = rig.leg;
  // torso: shorts at the hips, jersey over the trunk, number front and back
  add("pelvis", ellipsoid(0.055 * H, 0.095 * H, 0.05 * H, shorts), 0, 0, 0.005);
  add("spine", ellipsoid(0.062 * H, 0.10 * H, 0.10 * H, jersey), 0, 0, 0.0);
  add("chest", ellipsoid(0.068 * H, 0.12 * H, 0.09 * H, jersey), 0, 0, -0.04);
  const front = numberPlane(def.number, team, H, false); front.position.set(0.0692 * H, 0, -0.035 * H);
  const back = numberPlane(def.number, team, H, true); back.position.set(-0.0692 * H, 0, -0.035 * H);
  joints[index.chest].add(front, back);
  add("neck", (() => { const g = new THREE.Group(); const m = limb(0.03 * H, 0.06 * H, skin); m.position.z = 0.03 * H; g.add(m); return g; })());
  add("head", ellipsoid(0.062 * H, 0.062 * H, 0.071 * H, skin), 0, 0, 0.065);
  add("head", ellipsoid(0.066 * H, 0.066 * H, 0.05 * H, hair), 0, 0, 0.088);
  for (const s of ["L", "R"]) {
    add(`upper_arm_${s}`, limb(0.026 * H, aU * H, skin));
    add(`forearm_${s}`, limb(0.022 * H, aF * H, skin));
    const hand = ellipsoid(0.014 * H, 0.03 * H, 0.056 * H, skin);
    hand.position.set(0, 0, -0.052 * H);
    add(`hand_${s}`, hand);
    add(`clavicle_${s}`, ellipsoid(0.03 * H, 0.05 * H, 0.03 * H, jersey), 0, s === "L" ? 0.045 : -0.045, 0);
    const thigh = limb(0.042 * H, lT * H, skin);
    add(`thigh_${s}`, thigh);
    add(`thigh_${s}`, limb(0.05 * H, 0.12 * H, shorts));
    add(`shin_${s}`, limb(0.032 * H, lS * H, skin));
    const sh = new THREE.Mesh(new THREE.BoxGeometry(0.152 * H, 0.055 * H, 0.04 * H), shoe);
    sh.position.set(0.041 * H, 0, -0.019 * H);
    const so = new THREE.Mesh(new THREE.BoxGeometry(0.156 * H, 0.058 * H, 0.012 * H), sole);
    so.position.set(0.041 * H, 0, -0.037 * H);
    add(`foot_${s}`, sh); add(`foot_${s}`, so);
  }
  parentGroup.add(joints[0]);
  return {
    def, joints, pelvis: joints[0],
    pose(pelvisPos, q) {  // q: Float32Array view of 21*4 (x, y, z, w)
      joints[0].position.set(pelvisPos[0], pelvisPos[1], pelvisPos[2]);
      for (let j = 0; j < joints.length; j++) joints[j].quaternion.set(q[j * 4], q[j * 4 + 1], q[j * 4 + 2], q[j * 4 + 3]);
    },
  };
}

export function buildBall(parent, r) {
  const c = document.createElement("canvas");
  c.width = 512; c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#d9742b"; g.fillRect(0, 0, 512, 256);
  g.fillStyle = "rgba(255,255,255,0.05)";
  for (let i = 0; i < 400; i++) g.fillRect((i * 97) % 512, (i * 53) % 256, 2, 2);
  g.strokeStyle = "#1a1109"; g.lineWidth = 5;
  for (const x of [0, 128, 256, 384, 512]) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 256); g.stroke(); }
  g.beginPath(); g.moveTo(0, 128); g.lineTo(512, 128); g.stroke();
  for (const cx of [64, 192, 320, 448]) {
    g.beginPath(); g.ellipse(cx, 128, 64, 90, 0, 0, Math.PI * 2); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), new THREE.MeshStandardMaterial({ map: t, roughness: 0.65 }));
  m.castShadow = true;
  parent.add(m);
  return m;
}
