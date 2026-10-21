// Lob-pass player. Loads a baked anim.json and renders it; every pose, the ball and the director camera were
// computed in Python, so this file only plays them back (plus the net, the rim flex and the HUD).
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { createRenderer, createBackground, createLights, createCourt, createHoop, rimAngle } from "./scene.js";
import { Net } from "./net.js";
import { buildCharacter, buildBall } from "./figure.js";
import { Hud } from "./hud.js";

const params = new URLSearchParams(location.search);
const capture = params.has("capture");
if (capture) document.body.classList.add("capture");
if (params.has("nohud")) document.body.classList.add("nohud");

const f32 = (s) => {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
};
const toThree = (a, i = 0) => new THREE.Vector3(a[i], a[i + 2], -a[i + 1]);  // court (x, y, z up) -> three (x, z, -y)

async function main() {
  const data = await (await fetch(params.get("anim") || "anim.json")).json();
  const N = data.frames, K = data.constants;
  const simTime = f32(data.sim_time);
  const ballPos = f32(data.ball.pos), ballQ = f32(data.ball.quat);
  const camPos = f32(data.camera.pos), camTgt = f32(data.camera.target), camFov = f32(data.camera.fov);
  const chars = data.characters.map((c) => ({ def: c, pelvis: f32(c.pelvis), quats: f32(c.quats) }));

  const canvas = document.getElementById("c");
  const renderer = createRenderer(canvas, capture);
  const scene = new THREE.Scene();
  scene.background = createBackground();
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;  // court frame (Z up) -> three.js (Y up)
  scene.add(world);
  createLights(world);
  createCourt(world, K, data.court);
  const hoop = createHoop(world, K);
  const net = new Net(world, K.rim_z, K.ball_r);
  const figures = chars.map((c) => buildCharacter(c.def, data.rig, world));
  const ball = buildBall(world, K.ball_r);

  // overlays in the world: the dunker's planned path and the predicted drag-free arc
  const plan = data.overlay.plan_path.map((p) => new THREE.Vector3(p[0], p[1], 0.03));
  const pathLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(plan), new THREE.LineDashedMaterial({ color: 0xffe066, dashSize: 0.35, gapSize: 0.2, transparent: true, opacity: 0.85 }));
  pathLine.computeLineDistances();
  world.add(pathLine);
  const arcGeo = new THREE.BufferGeometry().setFromPoints(data.overlay.arc.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
  const arcPts = new THREE.Line(arcGeo, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.07, gapSize: 0.09, transparent: true, opacity: 0 }));
  arcPts.computeLineDistances();
  world.add(arcPts);

  const camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.1, 200);
  const controls = new OrbitControls(camera, canvas);
  controls.enabled = false;
  const hud = new Hud(data, simTime);
  const slam = data.events.find((e) => e.type === "slam");
  const slamT = slam ? simTime[Math.min(slam.frame, N - 1)] : null;
  const relT = data.overlay.release_frame != null ? simTime[data.overlay.release_frame] : null;

  function resize() {
    const w = capture ? 1920 : canvas.clientWidth, h = capture ? 1080 : canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // net physics follows the frame sequence (deterministic); scrubbing backwards replays from the start
  let netFrame = 0;
  const ballAt = (n) => [ballPos[n * 3], ballPos[n * 3 + 1], ballPos[n * 3 + 2]];
  function advanceNet(n) {
    if (n < netFrame) { net.reset(); netFrame = 0; }
    while (netFrame < n) {
      netFrame++;
      net.advance(simTime[netFrame] - simTime[netFrame - 1], ballAt(netFrame));
    }
  }

  let mode = "director";
  function setFrame(n) {
    n = Math.max(0, Math.min(N - 1, Math.round(n)));
    chars.forEach((c, i) => figures[i].pose(c.pelvis.subarray(n * 3, n * 3 + 3), c.quats.subarray(n * 84, n * 84 + 84)));
    ball.position.set(ballPos[n * 3], ballPos[n * 3 + 1], ballPos[n * 3 + 2]);
    ball.quaternion.set(ballQ[n * 4], ballQ[n * 4 + 1], ballQ[n * 4 + 2], ballQ[n * 4 + 3]);
    advanceNet(n);
    const t = simTime[n];
    hoop.rimPivot.rotation.y = slamT == null ? 0 : rimAngle(t - slamT);
    arcPts.material.opacity = relT == null || t < relT ? 0 : Math.max(0, 0.9 * (1 - (t - relT) / 1.2));
    if (mode === "director") {
      camera.position.copy(toThree(camPos, n * 3));
      camera.fov = camFov[n];
      camera.lookAt(toThree(camTgt, n * 3));
      camera.updateProjectionMatrix();
    } else {
      controls.update();
    }
    hud.draw(n);
    renderer.render(scene, camera);
    return n;
  }

  // ---- interactive controls
  let frame = 0, playing = !capture, last = performance.now(), speed = 1;
  const scrub = document.getElementById("scrub");
  scrub.max = String(N - 1);
  scrub.oninput = () => { frame = Number(scrub.value); setFrame(frame); };
  document.getElementById("play").onclick = (e) => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; };
  document.getElementById("speed").onchange = (e) => { speed = Number(e.target.value); };
  document.getElementById("camMode").onchange = (e) => {
    mode = e.target.value;
    controls.enabled = mode === "free";
    if (mode === "free") { camera.position.set(8, -12, 6).multiplyScalar(1); controls.target.set(2, 1, 0); controls.update(); }
  };
  if (!capture) {
    const loop = (now) => {
      const dt = (now - last) / 1000; last = now;
      if (playing) { frame += dt * data.fps * speed; if (frame >= N - 1) { frame = 0; net.reset(); netFrame = 0; } scrub.value = String(Math.floor(frame)); }
      setFrame(frame);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  // ---- deterministic capture API (frames advance only when asked)
  window.lobpass = {
    frames: N, fps: data.fps,
    setFrame: (n) => { setFrame(n); return n; },
    grab: (type = "image/png", q = 0.95) => canvas.toDataURL(type, q),
    ready: true,
  };
  setFrame(0);
}

main().catch((e) => { console.error(e); window.lobpassError = String(e); });
