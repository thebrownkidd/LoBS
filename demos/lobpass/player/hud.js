// HUD (B8): corner card with outcome badge, decision sparkline, release readout, outcome banner.
const LABEL = { DUNK: "DUNK!", MISS: "MISS", INTERCEPTED: "INTERCEPTED", BLOCKED: "BLOCKED", NO_PASS: "NO PASS", CATCH_NO_DUNK: "CAUGHT" };

export class Hud {
  constructor(data, simTime) {
    this.o = data.overlay;
    this.t = simTime;
    this.$ = (id) => document.getElementById(id);
    this.spark = this.$("spark").getContext("2d");
    this.$("stageLabel").textContent = `${this.o.stage}  ·  seed ${this.o.seed}`;
    const ck = String(this.o.checkpoint);
    this.$("subLabel").textContent = ck === "oracle" ? "scripted expert" : ck.split(/[\\/]/).slice(-1)[0].replace(".pt", "");
    // the frame where the outcome becomes known
    const ev = data.events;
    const first = (types) => ev.filter((e) => types.includes(e.type)).sort((a, b) => a.frame - b.frame)[0];
    const decisive = first(["slam"]) || first(["catch", "intercept", "rim_touch", "board_touch", "floor_touch"]);
    this.outcomeFrame = decisive ? decisive.frame : Math.max(data.frames - 90, 0);
    this.tEnd = simTime[simTime.length - 1];
    this.decisions = this.o.decisions;
    this.release = this.o.release_frame;
    const r = this.o.readout;
    if (r) {
      this.$("readout").innerHTML =
        `speed <b>${r.speed.toFixed(1)}</b> m/s<br>angle <b>${r.angle.toFixed(0)}</b>°` + (r.apex ? `<br>apex <b>${r.apex.toFixed(1)}</b> m` : "");
    }
  }
  draw(frame) {
    const showBadge = frame >= this.outcomeFrame;
    const badge = this.$("badge");
    badge.style.display = showBadge ? "block" : "none";
    badge.textContent = LABEL[this.o.outcome] || this.o.outcome;
    badge.className = this.o.outcome;
    const banner = this.$("banner");
    const k = frame - this.outcomeFrame;
    const a = k < 0 ? 0 : k < 12 ? k / 12 : k < 110 ? 1 : Math.max(0, 1 - (k - 110) / 25);
    banner.style.opacity = String(a * 0.95);
    banner.textContent = LABEL[this.o.outcome] || "";
    banner.style.color = this.o.outcome === "DUNK" ? "#27c46b" : this.o.outcome === "INTERCEPTED" || this.o.outcome === "BLOCKED" ? "#e5484d" : "#d0d5dd";
    banner.style.transform = `scale(${1 + 0.08 * Math.max(0, 1 - k / 14)})`;
    this.$("readout").style.display = this.release != null && frame >= this.release ? "block" : "none";
    // sparkline of P(throw now) over the episode, with the playhead
    const c = this.spark, W = 720, H = 112, now = this.t[Math.min(frame, this.t.length - 1)];
    const x = (t) => 8 + (t / this.tEnd) * (W - 16);
    const y = (p) => H - 10 - p * (H - 20);
    c.clearRect(0, 0, W, H);
    c.strokeStyle = "rgba(255,255,255,.18)"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(8, y(0.5)); c.lineTo(W - 8, y(0.5)); c.stroke();
    c.strokeStyle = "#7aa7ff"; c.lineWidth = 3; c.beginPath();
    const ds = this.decisions;
    ds.forEach((d, i) => (i ? c.lineTo(x(d.t), y(d.p)) : c.moveTo(x(d.t), y(d.p))));
    c.stroke();
    if (this.release != null) {
      const tr = this.t[this.release];
      c.strokeStyle = "#ff9a3c"; c.setLineDash([6, 5]); c.beginPath(); c.moveTo(x(tr), 4); c.lineTo(x(tr), H - 4); c.stroke(); c.setLineDash([]);
    }
    c.strokeStyle = "#ffffff"; c.lineWidth = 2; c.beginPath(); c.moveTo(x(now), 2); c.lineTo(x(now), H - 2); c.stroke();
  }
}
