// Native force laws, state-machine behaviors and the per-tick orchestration (mirrors lobs/forces.py,
// lobs/behavior.py and Simulation.step). Script behaviors are not supported here: simulations that have
// any stay on the Python runtime.
#pragma once
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <string>
#include <tuple>
#include <vector>
#include "kernel.hpp"
#include "program.hpp"

namespace lobs {

constexpr double kEps = 1e-9;

struct Act {
  int type = 0;  // 0 actuate 1 force 2 steer 3 set_var 4 emit 5 enable_force 6 enable_behavior
  int act_idx = -1, node = -1;
  Thing on;
  bool has_on = false;
  double max_force = 0;
  bool has_max = false;
  int mode = 0;  // 0 seek 1 arrive 2 flee 3 intercept 4 hold
  Thing target;
  bool has_target = false;
  double speed = 1, max_accel = 5, gain = 4, stop_radius = 0;
  bool plane = true;
  int var = -1, event = -1, idx = -1;
  bool flag = true;
};

struct StateDef {
  std::vector<Act> enter, run, exit;
};

struct Trans {
  int from = 0, to = 0, node = 0;
  double min_time = 0;
  int priority = 0;
};

struct HeldForce {
  Thing on;
  double v[3];
};

struct BehDef {
  std::string name;
  Thing on;
  bool has_on = false;
  int period = 1, latency = 0, initial = 0;
  bool enabled0 = true;
  std::vector<StateDef> states;
  std::vector<Trans> trans;
  std::vector<std::vector<int>> cand;  // per state: transition ids by (-priority, declaration)
  std::vector<int> var_n;
  std::vector<Val> var_init;
  // runtime
  std::vector<std::pair<int, double>> targets;  // held actuator targets
  std::vector<HeldForce> held;
};

struct LawDef {
  std::string name;
  Thing on, toward, axis_thing;
  bool has_toward = false, has_dir = false, has_axis = false, reaction = false, has_vector = false, has_when = false;
  double dir[3] = {0, 0, 0}, axis_local[3] = {0, 0, 0};
  int node_mag = -1, node_vec = -1, node_when = -1;
  bool has_peak = false, has_imp = false, has_act = false;
  double peak = 0, impulse = 0, active_time = 0;
};

struct LogEntry {
  double t;
  int beh, from, to;
};

struct Snap {
  std::vector<double> pos, quat, lin, ang;
  double t = 0;
  std::vector<Val> vals;
};

class Runtime {
 public:
  Program* prog = nullptr;
  Kernel* kernel = nullptr;
  nb::object err_cls;  // lobs.errors.LobsError
  double tick_dt = 0.01;
  int substeps = 1;
  std::vector<int> mj_body;  // registry body -> MuJoCo body id
  std::vector<int> owned;
  std::vector<BehDef> behs;
  std::vector<LawDef> laws;
  std::vector<LogEntry> log;
  std::vector<char> events_next;
  std::vector<WrenchRow> rows;
  Frame cur;
  std::vector<Val> cur_vals;
  std::deque<Snap> ring;
  int max_latency = 0;
  int tick = 0;

  [[noreturn]] void fail(const std::string& msg) {
    PyErr_SetString(err_cls.is_valid() ? err_cls.ptr() : PyExc_RuntimeError, msg.c_str());
    throw nb::python_error();
  }

  void finalize() {
    for (auto& b : behs) {
      b.cand.assign(b.states.size(), {});
      for (size_t i = 0; i < b.trans.size(); ++i) b.cand[b.trans[i].from].push_back(static_cast<int>(i));
      for (auto& c : b.cand)
        std::stable_sort(c.begin(), c.end(), [&](int x, int y) { return b.trans[x].priority > b.trans[y].priority; });
      max_latency = std::max(max_latency, b.latency);
    }
    prog->laws.assign(laws.size(), LawState());
    prog->behs.assign(behs.size(), BState());
  }

  void reset(const std::vector<std::tuple<int, int, std::vector<double>>>& var_overrides) {
    tick = 0;
    log.clear();
    ring.clear();
    std::fill(prog->events.begin(), prog->events.end(), 0);
    std::fill(events_next.begin(), events_next.end(), 0);
    for (auto& l : prog->laws) l = LawState();
    for (size_t i = 0; i < behs.size(); ++i) {
      BehDef& b = behs[i];
      BState& s = prog->behs[i];
      s = BState();
      s.state_idx = b.initial;
      s.enabled = b.enabled0;
      s.vars = b.var_init;
      b.targets.clear();
      b.held.clear();
    }
    for (auto& [bi, vi, vals] : var_overrides) {
      Val& v = prog->behs[bi].vars[vi];
      for (size_t k = 0; k < vals.size() && k < 4; ++k) v.v[k] = vals[k];
    }
  }

  // ---------------------------------------------------------------- helpers
  static double norm3(const double* a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

  static void unit3(const double* a, double* o) {
    double n = norm3(a);
    for (int c = 0; c < 3; ++c) o[c] = n > kEps ? a[c] / n : 0.0;
  }

  static void clip_norm3(const double* a, double limit, double* o) {
    double n = norm3(a);
    if (n > limit && n > kEps) {
      double f = limit / n;
      for (int c = 0; c < 3; ++c) o[c] = a[c] * f;
    } else {
      for (int c = 0; c < 3; ++c) o[c] = a[c];
    }
  }

  void add_row(int body, const double* f, const double* pt) {
    WrenchRow w;
    w.body = mj_body[body];
    for (int c = 0; c < 3; ++c) {
      w.f[c] = f[c];
      w.tau[c] = 0.0;
      w.pt[c] = pt[c];
    }
    rows.push_back(w);
  }

  // forces._apply: a world-frame force on an actor / body / anchor, at the current frame
  void apply_force(const Thing& th, const double* force) {
    if (th.kind == 1) {
      if (!prog->bodies[th.index].is_static) {
        double pc[3], vc[3];
        prog->query_com(th.index, pc, vc);
        add_row(th.index, force, pc);
      }
    } else if (th.kind == 2) {
      std::vector<int> bs;
      double total = 0.0;
      for (int i : prog->actors[th.index].bodies) {
        const BodyInfo& bi = prog->bodies[i];
        if (!bi.is_static && bi.mass > 0) bs.push_back(i);
      }
      for (int i : bs) total += prog->bodies[i].mass;
      for (int i : bs) {
        double f[3], pc[3], vc[3];
        for (int c = 0; c < 3; ++c) f[c] = force[c] * prog->bodies[i].mass / total;
        prog->query_com(i, pc, vc);
        add_row(i, f, pc);
      }
    } else if (th.kind == 3) {
      int parent = prog->anchors[th.index].parent;
      if (parent >= 0 && !prog->bodies[parent].is_static) {
        double p[3];
        prog->query_pos(th, p);
        add_row(parent, force, p);
      }
    }
  }

  double mass_of(const Thing& th) {
    double m = 0.0;
    if (th.kind == 2) {
      for (int i : prog->actors[th.index].bodies)
        if (!prog->bodies[i].is_static) m += prog->bodies[i].mass;
    } else if (th.kind == 1 && !prog->bodies[th.index].is_static) {
      m = prog->bodies[th.index].mass;
    }
    if (m <= 0) fail("cannot steer: the actor is static or has no mass");
    return m;
  }

  // ---------------------------------------------------------------- behaviors
  struct Cmd {
    std::vector<std::pair<int, double>> targets;
    std::vector<HeldForce> forces, once;
    std::vector<int> events;
    std::vector<std::pair<int, bool>> force_switch, beh_switch;
  };

  const Val& val(const std::vector<Val>& vals, int node) { return vals[node]; }

  void run_actions(int bi, const std::vector<Act>& acts, bool hold, const std::vector<Val>& vals, Cmd& cmd) {
    BehDef& b = behs[bi];
    for (const Act& a : acts) {
      switch (a.type) {
        case 0: {
          double v = vals[a.node].v[0];
          if (!std::isfinite(v)) fail("behavior '" + b.name + "': non-finite actuator command");
          bool found = false;
          for (auto& pr : cmd.targets)
            if (pr.first == a.act_idx) { pr.second = v; found = true; }
          if (!found) cmd.targets.emplace_back(a.act_idx, v);
          break;
        }
        case 1: {
          const Val& vec = vals[a.node];
          double v[3] = {vec.v[0], vec.v[1], vec.v[2]};
          if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]))
            fail("behavior '" + b.name + "': force must be a finite 3-vector");
          if (a.has_max) clip_norm3(v, a.max_force, v);
          HeldForce hf{a.has_on ? a.on : b.on, {v[0], v[1], v[2]}};
          (hold ? cmd.forces : cmd.once).push_back(hf);
          break;
        }
        case 2: steer(b, a, hold, cmd); break;
        case 3: {
          const Val& v = vals[a.node];
          Val& dst = prog->behs[bi].vars[a.var];
          for (int k = 0; k < dst.n; ++k) {
            if (!std::isfinite(v.v[k])) fail("behavior '" + b.name + "': non-finite value for a variable");
            dst.v[k] = v.v[k];
          }
          break;
        }
        case 4: cmd.events.push_back(a.event); break;
        case 5: cmd.force_switch.emplace_back(a.idx, a.flag); break;
        case 6: cmd.beh_switch.emplace_back(a.idx, a.flag); break;
      }
    }
  }

  void steer(BehDef& b, const Act& a, bool hold, Cmd& cmd) {
    const Thing me = a.has_on ? a.on : b.on;
    const double mass = mass_of(me);
    double p[3], v[3], w[3];
    prog->query_pos(me, p);
    prog->query_twist(me, v, w);
    double vdes[3] = {0, 0, 0};
    if (a.mode != 4) {
      double pt[3], vt[3], wt[3];
      prog->query_pos(a.target, pt);
      prog->query_twist(a.target, vt, wt);
      if (a.plane) { pt[2] = p[2]; vt[2] = 0.0; }
      double dv[3] = {pt[0] - p[0], pt[1] - p[1], pt[2] - p[2]};
      double dist = norm3(dv), u[3];
      if (a.mode == 0) {
        unit3(dv, u);
        for (int c = 0; c < 3; ++c) vdes[c] = a.speed * u[c];
      } else if (a.mode == 2) {
        unit3(dv, u);
        for (int c = 0; c < 3; ++c) vdes[c] = -a.speed * u[c];
      } else if (a.mode == 1) {
        double room = std::max(dist - a.stop_radius, 0.0);
        double s = std::min(a.speed, 0.8 * std::sqrt(2.0 * a.max_accel * room));
        unit3(dv, u);
        for (int c = 0; c < 3; ++c) vdes[c] = s * u[c];
      } else {
        double horizon = a.speed > kEps ? dist / a.speed : 0.0;
        double aim[3] = {pt[0] + vt[0] * horizon - p[0], pt[1] + vt[1] * horizon - p[1], pt[2] + vt[2] * horizon - p[2]};
        unit3(aim, u);
        for (int c = 0; c < 3; ++c) vdes[c] = a.speed * u[c];
      }
    }
    if (a.plane) { vdes[2] = 0.0; v[2] = 0.0; }
    double e[3] = {a.gain * (vdes[0] - v[0]), a.gain * (vdes[1] - v[1]), a.gain * (vdes[2] - v[2])}, acc[3];
    clip_norm3(e, a.max_accel, acc);
    if (a.plane) acc[2] = 0.0;
    HeldForce hf{me, {mass * acc[0], mass * acc[1], mass * acc[2]}};
    (hold ? cmd.forces : cmd.once).push_back(hf);
  }

  void decide(int bi, double t, const std::vector<Val>& vals) {
    BehDef& b = behs[bi];
    BState& s = prog->behs[bi];
    Cmd cmd;
    if (!s.entered) {
      s.entered = true;
      s.enter_t = t;
      run_actions(bi, b.states[s.state_idx].enter, false, vals, cmd);
    }
    for (int ti : b.cand[s.state_idx]) {
      const Trans& tr = b.trans[ti];
      if (t - s.enter_t >= tr.min_time - kEps && vals[tr.node].v[0] != 0.0) {
        run_actions(bi, b.states[s.state_idx].exit, false, vals, cmd);
        log.push_back({t, bi, s.state_idx, tr.to});
        s.state_idx = tr.to;
        s.enter_t = t;
        run_actions(bi, b.states[s.state_idx].enter, false, vals, cmd);
        break;
      }
    }
    run_actions(bi, b.states[s.state_idx].run, true, vals, cmd);
    for (auto& pr : cmd.targets) {
      bool found = false;
      for (auto& q : b.targets) if (q.first == pr.first) { q.second = pr.second; found = true; }
      if (!found) b.targets.push_back(pr);
    }
    b.held = std::move(cmd.forces);
    for (auto& hf : cmd.once) pending_once.push_back(hf);
    for (int e : cmd.events) events_next[e] = 1;
    for (auto& sw : cmd.force_switch) force_sw.push_back(sw);
    for (auto& sw : cmd.beh_switch) beh_sw.push_back(sw);
  }

  std::vector<HeldForce> pending_once;
  std::vector<std::pair<int, bool>> force_sw, beh_sw;

  void behaviors_step(double t) {
    std::fill(events_next.begin(), events_next.end(), 0);
    force_sw.clear();
    beh_sw.clear();
    // the current frame is what force application and latency-0 decisions use
    const double *cp = cur.pos.data(), *cq = cur.quat.data(), *cl = cur.lin.data(), *ca = cur.ang.data();
    for (size_t bi = 0; bi < behs.size(); ++bi) {
      BehDef& b = behs[bi];
      BState& s = prog->behs[bi];
      if (!s.enabled) {
        b.held.clear();
        continue;
      }
      if (tick % b.period == 0) {
        pending_once.clear();
        const std::vector<Val>* vals = &cur_vals;
        const Snap* snap = nullptr;
        if (b.latency > 0) {
          const int n = static_cast<int>(ring.size());
          snap = n > b.latency ? &ring[n - 1 - b.latency] : &ring[0];
          vals = &snap->vals;
          prog->use_frame(snap->pos.data(), snap->quat.data(), snap->lin.data(), snap->ang.data(), snap->t);
        }
        decide(static_cast<int>(bi), t, *vals);
        if (snap) prog->use_frame(cp, cq, cl, ca, t);
        for (auto& hf : pending_once) apply_force(hf.on, hf.v);
      }
      for (auto& hf : b.held) apply_force(hf.on, hf.v);
    }
    for (auto& sw : force_sw) prog->laws[sw.first].enabled = sw.second;
    for (auto& sw : beh_sw) prog->behs[sw.first].enabled = sw.second;
    prog->events = events_next;
    for (auto& b : behs)
      for (auto& pr : b.targets) kernel->targets[pr.first] = pr.second;
  }

  // ---------------------------------------------------------------- force laws
  static void limit(const LawDef& L, const LawState& st, double* f, double dt) {
    if (L.has_peak) {
      double n = norm3(f);
      if (n > L.peak && n > 1e-12) {
        double k = L.peak / n;
        for (int c = 0; c < 3; ++c) f[c] *= k;
      }
    }
    if (L.has_act && st.active_time >= L.active_time - 1e-9) {
      f[0] = f[1] = f[2] = 0.0;
      return;
    }
    if (L.has_imp) {
      double remaining = L.impulse - st.impulse;
      if (remaining <= 1e-12) {
        f[0] = f[1] = f[2] = 0.0;
        return;
      }
      double need = norm3(f) * dt;
      if (need > remaining) {
        double k = remaining / need;
        for (int c = 0; c < 3; ++c) f[c] *= k;
      }
    }
  }

  void laws_apply(double t) {
    for (size_t li = 0; li < laws.size(); ++li) {
      const LawDef& L = laws[li];
      LawState& st = prog->laws[li];
      bool on = st.enabled && (!L.has_when || cur_vals[L.node_when].v[0] != 0.0);
      if (!on) {
        st.active = false;
        st.last[0] = st.last[1] = st.last[2] = 0.0;
        continue;
      }
      double f[3];
      if (L.has_vector) {
        const Val& v = cur_vals[L.node_vec];
        for (int c = 0; c < 3; ++c) f[c] = v.v[c];
        if (!std::isfinite(f[0]) || !std::isfinite(f[1]) || !std::isfinite(f[2]))
          fail("force '" + L.name + "' produced a non-finite vector at t=" + std::to_string(t));
      } else {
        double mag = cur_vals[L.node_mag].v[0];
        if (!std::isfinite(mag)) fail("force '" + L.name + "' produced a non-finite magnitude at t=" + std::to_string(t));
        double unit[3];
        if (L.has_toward) {
          double pa[3], pb[3];
          prog->query_pos(L.toward, pa);
          prog->query_pos(L.on, pb);
          double dv[3] = {pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]};
          double n = norm3(dv);
          if (n < 1e-12) {
            st.active = false;
            st.last[0] = st.last[1] = st.last[2] = 0.0;
            continue;
          }
          for (int c = 0; c < 3; ++c) unit[c] = dv[c] / n;
        } else if (L.has_dir) {
          for (int c = 0; c < 3; ++c) unit[c] = L.dir[c];
        } else {
          double R[9];
          prog->query_R(L.axis_thing, R);
          for (int r = 0; r < 3; ++r) unit[r] = R[3 * r] * L.axis_local[0] + R[3 * r + 1] * L.axis_local[1] + R[3 * r + 2] * L.axis_local[2];
        }
        for (int c = 0; c < 3; ++c) f[c] = mag * unit[c];
      }
      limit(L, st, f, tick_dt);
      double fmag = norm3(f);
      if (fmag == 0.0) {
        st.active = false;
        for (int c = 0; c < 3; ++c) st.last[c] = f[c];
        continue;
      }
      st.active = true;
      for (int c = 0; c < 3; ++c) st.last[c] = f[c];
      st.impulse += fmag * tick_dt;
      st.active_time += tick_dt;
      st.peak = std::max(st.peak, fmag);
      apply_force(L.on, f);
      if (L.reaction) {
        double g[3] = {-f[0], -f[1], -f[2]};
        apply_force(L.toward, g);
      }
    }
  }

  // ---------------------------------------------------------------- observation / tick
  void snapshot_current(double t) {
    if (max_latency == 0) return;
    Snap s;
    s.pos = cur.pos; s.quat = cur.quat; s.lin = cur.lin; s.ang = cur.ang;
    s.t = t;
    s.vals = cur_vals;
    ring.push_back(std::move(s));
    while (static_cast<int>(ring.size()) > max_latency + 1) ring.pop_front();
  }

  void observe(const Frame& f, double t) {
    cur = f;
    prog->begin_tick_raw(cur.pos.data(), cur.quat.data(), cur.lin.data(), cur.ang.data(), cur.q.data(), cur.qd.data(),
                         cur.eff.data(), t);
    prog->eval_nodes();
    cur_vals = prog->values();
  }

  // One Simulation.step: behaviors, force laws, user wrenches, physics, new observation.
  void step_tick(const std::vector<WrenchRow>& user_rows) {
    const double t = tick * tick_dt;
    rows.clear();
    // current-tick snapshot for latency readers (before any decision this tick)
    snapshot_current(t);
    if (!behs.empty()) behaviors_step(t);
    if (!laws.empty()) laws_apply(t);
    for (const auto& w : user_rows) rows.push_back(w);
    kernel->step_rows(substeps, rows);
    ++tick;
    Frame f;
    kernel->read_raw(f);
    observe(f, tick * tick_dt);
  }
};

}  // namespace lobs
