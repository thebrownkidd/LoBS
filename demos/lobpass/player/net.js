// The net: a small deterministic Verlet cloth. 12 strands x 6 knots, 0.42 m long, top ring r 0.2286, bottom ring
// r 0.14, 4 constraint iterations, damping 0.985, collides with the ball sphere. Fixed 120 Hz substeps, no
// randomness, so the same frame sequence always produces the same net.
import * as THREE from "three";

const STRANDS = 12, KNOTS = 6, LENGTH = 0.42;
const TOP_R = 0.2286, BOT_R = 0.14, DAMP = 0.985, ITER = 4, SUB = 1 / 120, GRAV = 9.81 * 0.55;

export class Net {
  constructor(parent, rimZ, ballR) {
    this.rimZ = rimZ; this.ballR = ballR + 0.01;
    this.n = STRANDS * KNOTS;
    this.p = new Float32Array(this.n * 3); this.q = new Float32Array(this.n * 3);
    this.rest = [];  // [a, b, length]
    for (let i = 0; i < STRANDS; i++) {
      for (let k = 0; k < KNOTS; k++) {
        const r = TOP_R + (BOT_R - TOP_R) * (k / (KNOTS - 1));
        if (k < KNOTS - 1) this.rest.push([this.idx(i, k), this.idx(i, k + 1), LENGTH / (KNOTS - 1)]);
        const rr = (2 * Math.PI * r) / STRANDS;
        this.rest.push([this.idx(i, k), this.idx((i + 1) % STRANDS, k), rr * 0.98]);
      }
    }
    const pos = new Float32Array(this.rest.length * 6);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.mesh = new THREE.LineSegments(this.geo, new THREE.LineBasicMaterial({ color: 0xf2f2f2, transparent: true, opacity: 0.9 }));
    this.mesh.frustumCulled = false;
    parent.add(this.mesh);
    this.reset();
  }
  idx(i, k) { return i * KNOTS + k; }
  reset() {
    for (let i = 0; i < STRANDS; i++) {
      const a = (i / STRANDS) * 2 * Math.PI;
      for (let k = 0; k < KNOTS; k++) {
        const r = TOP_R + (BOT_R - TOP_R) * (k / (KNOTS - 1));
        const j = this.idx(i, k) * 3;
        this.p[j] = r * Math.cos(a); this.p[j + 1] = r * Math.sin(a); this.p[j + 2] = this.rimZ - (LENGTH * k) / (KNOTS - 1);
      }
    }
    this.q.set(this.p);
    this.acc = 0;
    this.write();
  }
  // advance by ``dt`` seconds of simulated time with the ball at ``ball`` (court frame)
  advance(dt, ball) {
    this.acc += Math.min(dt, 0.1);
    while (this.acc >= SUB - 1e-9) { this.acc -= SUB; this.step(ball); }
    this.write();
  }
  step(ball) {
    const { p, q } = this;
    for (let i = 0; i < this.n; i++) {
      if (i % KNOTS === 0) continue;  // top knots are pinned to the rim
      const j = i * 3;
      for (let d = 0; d < 3; d++) {
        const cur = p[j + d], vel = (cur - q[j + d]) * DAMP;
        q[j + d] = cur;
        p[j + d] = cur + vel + (d === 2 ? -GRAV * SUB * SUB : 0);
      }
    }
    for (let it = 0; it < ITER; it++) {
      for (const [a, b, L] of this.rest) {
        const ja = a * 3, jb = b * 3;
        const dx = p[jb] - p[ja], dy = p[jb + 1] - p[ja + 1], dz = p[jb + 2] - p[ja + 2];
        const d = Math.hypot(dx, dy, dz) || 1e-9;
        const diff = (d - L) / d;
        const wa = a % KNOTS === 0 ? 0 : 0.5, wb = b % KNOTS === 0 ? 0 : 0.5;
        const wsum = wa + wb || 1;
        const fa = wa / wsum, fb = wb / wsum;
        p[ja] += dx * diff * fa; p[ja + 1] += dy * diff * fa; p[ja + 2] += dz * diff * fa;
        p[jb] -= dx * diff * fb; p[jb + 1] -= dy * diff * fb; p[jb + 2] -= dz * diff * fb;
      }
      if (ball) {
        for (let i = 0; i < this.n; i++) {
          if (i % KNOTS === 0) continue;
          const j = i * 3;
          const dx = p[j] - ball[0], dy = p[j + 1] - ball[1], dz = p[j + 2] - ball[2];
          const d = Math.hypot(dx, dy, dz);
          if (d < this.ballR && d > 1e-6) {
            const s = this.ballR / d;
            p[j] = ball[0] + dx * s; p[j + 1] = ball[1] + dy * s; p[j + 2] = ball[2] + dz * s;
          }
        }
      }
      // pin the top ring
      for (let i = 0; i < STRANDS; i++) {
        const a = (i / STRANDS) * 2 * Math.PI, j = this.idx(i, 0) * 3;
        p[j] = TOP_R * Math.cos(a); p[j + 1] = TOP_R * Math.sin(a); p[j + 2] = this.rimZ;
      }
    }
  }
  write() {
    const out = this.geo.attributes.position.array;
    let o = 0;
    for (const [a, b] of this.rest) {
      out[o++] = this.p[a * 3]; out[o++] = this.p[a * 3 + 1]; out[o++] = this.p[a * 3 + 2];
      out[o++] = this.p[b * 3]; out[o++] = this.p[b * 3 + 1]; out[o++] = this.p[b * 3 + 2];
    }
    this.geo.attributes.position.needsUpdate = true;
  }
}
