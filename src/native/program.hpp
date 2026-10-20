// Native measure program: an interpreter over the typed node list the Python compiler produces.
// Nodes it cannot express natively are evaluated by calling back into Python (their original closure), so
// every measure works; the native subset exists for speed. Semantics mirror lobs/measure/compile.py.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace lobs {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;

enum Op : int {
  CONST = 0, TIME, ADD, SUB, MUL, DIV, POW, NEG, ABS, LT, LE, GT, GE, EQ, AND, OR, NOT, INDEX, NORM, DOT, CROSS,
  SIN, COS, CLIP, WHERE, MAXOF, MINOF, MEANOF, SUMOF, VEC, UNIT, CLIPNORM, COPY,
  DERIV, INTEGRAL, RUNMAX, RUNMIN, ATRESET, SINCERESET, WINDOW, DURTRUE, TIMESINCE, RISING, FALLING, LATCH,
  HOLD, COOLDOWN, ONCE, SAMPLEON,
  P_POS, P_VEL, P_SPEED, P_HEIGHT, P_ORIENT, P_ANGVEL, P_SPIN, P_JQ, P_JQD, P_JEFF, REL_DIST,
  P_EVENT, P_BSTATE, P_BINSTATE, P_BTIME, P_BVAR, P_FIMPULSE, P_FACTIVETIME, P_FACTIVE, P_FAPPLIED,
  CALLBACK = 1000  // never executed natively: such nodes stay in Python
};

struct Val {
  double v[4] = {0, 0, 0, 0};
  int n = 1;
};

struct NodeDesc {
  int op = CONST;
  int kind = 0;   // 0 real, 1 bool, 2 quat
  int n = 1;      // elements
  int ndim = 0;   // 0 scalar shape (), 1 vector shape (n,)
  std::vector<int> in;
  double p[4] = {0, 0, 0, 0};
  int i[4] = {0, 0, 0, 0};
  bool eager = false;
};

struct Thing {
  int kind = 0;  // 0 world, 1 body, 2 actor, 3 anchor
  int index = -1;
  bool com = true;
};

struct BodyInfo {
  bool is_static = true;
  int dyn = -1;
  double wp[3] = {0, 0, 0}, wq[4] = {1, 0, 0, 0}, com[3] = {0, 0, 0};
  double mass = 0;
};

struct ActorInfo {
  std::vector<int> bodies;
  double mass = 0;
};

struct AnchorInfo {
  int parent = -1;
  double pos[3] = {0, 0, 0}, quat[4] = {1, 0, 0, 0};
  int rule = 0;  // 0 rigid, 1 translate_only
};

// Live force-law and behavior state that measures may read (owned by the native runtime, see runtime.hpp)
struct LawState {
  bool enabled = true, active = false;
  double impulse = 0, active_time = 0, peak = 0;
  double last[3] = {0, 0, 0};
};

struct BState {
  int state_idx = 0;
  double enter_t = 0;
  bool entered = false, enabled = true;
  std::vector<Val> vars;
};

struct NodeState {
  double a[4] = {0, 0, 0, 0};
  double b[4] = {0, 0, 0, 0};
  double t = 0, last = 0;
  bool has = false, prev = false, flag = false, has_last = false;
  std::deque<std::pair<double, Val>> buf;
};

struct Pose {
  double p[3], R[9], q[4];
};

class Program {
 public:
  std::vector<NodeDesc> nodes;
  std::vector<std::vector<Thing>> things;  // per node, the things its prop / rel refers to
  std::vector<BodyInfo> bodies;
  std::vector<ActorInfo> actors;
  std::vector<AnchorInfo> anchors;
  std::vector<int> order;  // native nodes, ascending
  std::vector<LawState> laws;
  std::vector<BState> behs;
  std::vector<char> events;  // events emitted by the last behavior step, by event id
  // frame (valid during a tick)
  const double *pos = nullptr, *quat = nullptr, *lin = nullptr, *ang = nullptr, *jq = nullptr, *jqd = nullptr, *jeff = nullptr;
  double t = 0;

  void init(size_t n) {
    nodes.assign(n, NodeDesc());
    things.assign(n, {});
    st_.assign(n, NodeState());
    val_.assign(n, Val());
  }

  void finalize() {
    pgen_.assign(bodies.size(), 0);
    cgen_.assign(bodies.size(), 0);
    ppose_.assign(bodies.size(), Pose());
    pcom_p_.assign(bodies.size(), {0, 0, 0});
    pcom_v_.assign(bodies.size(), {0, 0, 0});
  }

  void reset() {
    for (auto& s : st_) s = NodeState();
  }

  void begin_tick_raw(const double* p, const double* q, const double* l, const double* a, const double* jqa,
                     const double* jqda, const double* jea, double time) {
    pos = p; quat = q; lin = l; ang = a; jq = jqa; jqd = jqda; jeff = jea; t = time;
    ++tick_;
  }

  void begin_tick(nb::ndarray<const double, nb::c_contig> p, nb::ndarray<const double, nb::c_contig> q,
                  nb::ndarray<const double, nb::c_contig> l, nb::ndarray<const double, nb::c_contig> a,
                  nb::ndarray<const double, nb::c_contig> jqa, nb::ndarray<const double, nb::c_contig> jqda,
                  nb::ndarray<const double, nb::c_contig> jea, double time) {
    frame_keep_ = nb::make_tuple(p, q, l, a, jqa, jqda, jea);  // keep the arrays alive while pointers are used
    pos = p.data(); quat = q.data(); lin = l.data(); ang = a.data();
    jq = jqa.data(); jqd = jqda.data(); jeff = jea.data();
    t = time;
    ++tick_;
  }

  // Point the scene queries at another frame (latency snapshots); caches are invalidated.
  void use_frame(const double* p, const double* q, const double* l, const double* a, double time) {
    pos = p; quat = q; lin = l; ang = a; t = time;
    ++tick_;
  }

  // scene queries for the runtime (force application and steering)
  void query_pos(const Thing& th, double* p) { thing_pos(th, p); }
  void query_twist(const Thing& th, double* v, double* w) { thing_twist(th, v, w); }
  void query_com(int b, double* pc, double* vc) { body_com(b, pc, vc); }
  const Pose& query_pose(int b) { return body_pose(b); }
  void query_anchor_pose(int ai, double* p, double* q) { anchor_pose(ai, p, q); }
  void query_R(const Thing& th, double* R) {
    switch (th.kind) {
      case 0:
        for (int k = 0; k < 9; ++k) R[k] = (k % 4 == 0) ? 1.0 : 0.0;
        return;
      case 1: { const Pose& P = body_pose(th.index); for (int k = 0; k < 9; ++k) R[k] = P.R[k]; return; }
      case 2: { const Pose& P = body_pose(actors[th.index].bodies[0]); for (int k = 0; k < 9; ++k) R[k] = P.R[k]; return; }
      default: { double p[3], q[4]; anchor_pose(th.index, p, q); quat_to_R(q, R); }
    }
  }

  // Evaluate every native node, in order (inputs always have smaller ids), and return all values as an
  // (N, 4) array copy: row i holds node i (unused rows are zero). Callback-free by design.
  void eval_nodes() {
    for (int i : order) {
      Val out;
      out.n = nodes[i].n;
      exec(i, nodes[i], out);
      val_[i] = out;
    }
  }

  const std::vector<Val>& values() const { return val_; }

  nb::object values_array() {
    const size_t N = nodes.size();
    auto* buf = new double[N * 4]();
    for (int i : order) std::memcpy(buf + 4 * i, val_[i].v, 4 * sizeof(double));
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    size_t shp[2] = {N, 4};
    return nb::cast(nb::ndarray<nb::numpy, double>(buf, 2, shp, owner));
  }

  nb::object eval_all() {
    eval_nodes();
    return values_array();
  }

 private:
  uint64_t tick_ = 0;
  std::vector<NodeState> st_;
  std::vector<Val> val_;
  std::vector<uint64_t> pgen_, cgen_;
  std::vector<Pose> ppose_;
  std::vector<std::array<double, 3>> pcom_p_, pcom_v_;
  nb::object frame_keep_;

  // ---------------------------------------------------------------- scene
  static void quat_to_R(const double* qin, double* R) {
    double w = qin[0], x = qin[1], y = qin[2], z = qin[3];
    double n = std::sqrt(w * w + x * x + y * y + z * z);
    w /= n; x /= n; y /= n; z /= n;
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
  }

  static void quat_mul(const double* a, const double* b, double* o) {
    o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  }

  static void matvec(const double* R, const double* v, double* o) {
    o[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    o[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    o[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
  }

  static void cross3(const double* a, const double* b, double* o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
  }

  const Pose& body_pose(int b) {
    if (pgen_[b] == tick_) return ppose_[b];
    const BodyInfo& bi = bodies[b];
    Pose& P = ppose_[b];
    const double* q = bi.is_static ? bi.wq : quat + 4 * bi.dyn;
    const double* p = bi.is_static ? bi.wp : pos + 3 * bi.dyn;
    for (int c = 0; c < 3; ++c) P.p[c] = p[c];
    for (int c = 0; c < 4; ++c) P.q[c] = q[c];
    quat_to_R(q, P.R);
    pgen_[b] = tick_;
    return P;
  }

  void body_twist(int b, double* v, double* w) {
    const BodyInfo& bi = bodies[b];
    if (bi.is_static) {
      v[0] = v[1] = v[2] = w[0] = w[1] = w[2] = 0.0;
      return;
    }
    for (int c = 0; c < 3; ++c) {
      v[c] = lin[3 * bi.dyn + c];
      w[c] = ang[3 * bi.dyn + c];
    }
  }

  // COM position and velocity of a body
  void body_com(int b, double* pc, double* vc) {
    if (cgen_[b] != tick_) {
      const Pose& P = body_pose(b);
      double v[3], w[3], rc[3], wxr[3];
      body_twist(b, v, w);
      matvec(P.R, bodies[b].com, rc);
      cross3(w, rc, wxr);
      for (int c = 0; c < 3; ++c) {
        pcom_p_[b][c] = P.p[c] + rc[c];
        pcom_v_[b][c] = v[c] + wxr[c];
      }
      cgen_[b] = tick_;
    }
    for (int c = 0; c < 3; ++c) {
      pc[c] = pcom_p_[b][c];
      vc[c] = pcom_v_[b][c];
    }
  }

  void anchor_pose(int ai, double* p, double* q) {
    const AnchorInfo& a = anchors[ai];
    double pb[3] = {0, 0, 0}, qb[4] = {1, 0, 0, 0}, R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (a.parent >= 0) {
      const Pose& P = body_pose(a.parent);
      for (int c = 0; c < 3; ++c) pb[c] = P.p[c];
      for (int c = 0; c < 4; ++c) qb[c] = P.q[c];
      for (int c = 0; c < 9; ++c) R[c] = P.R[c];
    }
    double off[3];
    matvec(R, a.pos, off);
    for (int c = 0; c < 3; ++c) p[c] = pb[c] + off[c];
    if (a.rule == 1) {
      for (int c = 0; c < 4; ++c) q[c] = a.quat[c];
    } else {
      quat_mul(qb, a.quat, q);
    }
  }

  // position of a thing (point = COM by default, origin when th.com is false)
  void thing_pos(const Thing& th, double* p) {
    switch (th.kind) {
      case 0: p[0] = p[1] = p[2] = 0.0; return;
      case 1: {
        const Pose& P = body_pose(th.index);
        if (th.com) {
          double rc[3];
          matvec(P.R, bodies[th.index].com, rc);
          for (int c = 0; c < 3; ++c) p[c] = P.p[c] + rc[c];
        } else {
          for (int c = 0; c < 3; ++c) p[c] = P.p[c];
        }
        return;
      }
      case 2: {
        const ActorInfo& A = actors[th.index];
        int root = A.bodies[0];
        if (th.com && A.mass > 0) {
          if (A.bodies.size() == 1) {
            double pc[3], vc[3];
            body_com(root, pc, vc);
            for (int c = 0; c < 3; ++c) p[c] = pc[c];
          } else {
            double s[3] = {0, 0, 0};
            for (int b : A.bodies) {
              double pc[3], vc[3];
              body_com(b, pc, vc);
              for (int c = 0; c < 3; ++c) s[c] += bodies[b].mass * pc[c];
            }
            for (int c = 0; c < 3; ++c) p[c] = s[c] / A.mass;
          }
        } else {
          const Pose& P = body_pose(root);
          for (int c = 0; c < 3; ++c) p[c] = P.p[c];
        }
        return;
      }
      default: {
        double q[4];
        anchor_pose(th.index, p, q);
        return;
      }
    }
  }

  // velocity (and angular velocity) of a thing's point
  void thing_twist(const Thing& th, double* v, double* w) {
    switch (th.kind) {
      case 0: v[0] = v[1] = v[2] = w[0] = w[1] = w[2] = 0.0; return;
      case 1: {
        body_twist(th.index, v, w);
        if (th.com) {
          double pc[3], vc[3];
          body_com(th.index, pc, vc);
          for (int c = 0; c < 3; ++c) v[c] = vc[c];
        }
        return;
      }
      case 2: {
        const ActorInfo& A = actors[th.index];
        int root = A.bodies[0];
        body_twist(root, v, w);
        if (th.com && A.mass > 0) {
          if (A.bodies.size() == 1) {
            double pc[3], vc[3];
            body_com(root, pc, vc);
            for (int c = 0; c < 3; ++c) v[c] = vc[c];
          } else {
            double s[3] = {0, 0, 0};
            for (int b : A.bodies) {
              double pc[3], vc[3];
              body_com(b, pc, vc);
              for (int c = 0; c < 3; ++c) s[c] += bodies[b].mass * vc[c];
            }
            for (int c = 0; c < 3; ++c) v[c] = s[c] / A.mass;
          }
        }
        return;
      }
      default: {
        const AnchorInfo& a = anchors[th.index];
        if (a.parent < 0 || bodies[a.parent].is_static) {
          v[0] = v[1] = v[2] = w[0] = w[1] = w[2] = 0.0;
          return;
        }
        const Pose& P = body_pose(a.parent);
        double vb[3], wb[3], off[3], wxo[3];
        body_twist(a.parent, vb, wb);
        matvec(P.R, a.pos, off);
        cross3(wb, off, wxo);
        for (int c = 0; c < 3; ++c) v[c] = vb[c] + wxo[c];
        for (int c = 0; c < 3; ++c) w[c] = a.rule == 1 ? 0.0 : wb[c];
      }
    }
  }

  void thing_quat(const Thing& th, double* q) {
    switch (th.kind) {
      case 0: q[0] = 1; q[1] = q[2] = q[3] = 0; return;
      case 1: { const Pose& P = body_pose(th.index); for (int c = 0; c < 4; ++c) q[c] = P.q[c]; return; }
      case 2: { const Pose& P = body_pose(actors[th.index].bodies[0]); for (int c = 0; c < 4; ++c) q[c] = P.q[c]; return; }
      default: { double p[3]; anchor_pose(th.index, p, q); }
    }
  }

  // ---------------------------------------------------------------- execution
  static double wrap_angle(double a) { return a - 2.0 * kPi * std::ceil((a - kPi) / (2.0 * kPi)); }

  static double broadcast(const Val& a, int k) { return a.n == 1 ? a.v[0] : a.v[k]; }

  void exec(int id, const NodeDesc& d, Val& o) {
    NodeState& s = st_[id];
    auto in = [&](int k) -> const Val& { return val_[d.in[k]]; };
    const int n = d.n;
    switch (d.op) {
      case CONST: for (int k = 0; k < n; ++k) o.v[k] = d.p[k]; return;
      case TIME: o.v[0] = t; return;
      case ADD: case SUB: case MUL: case DIV: case LT: case LE: case GT: case GE: case EQ: case AND: case OR: {
        const Val &a = in(0), &b = in(1);
        for (int k = 0; k < n; ++k) {
          double x = broadcast(a, k), y = broadcast(b, k), r = 0;
          switch (d.op) {
            case ADD: r = x + y; break;
            case SUB: r = x - y; break;
            case MUL: r = x * y; break;
            case DIV: r = x / y; break;
            case LT: r = x < y; break;
            case LE: r = x <= y; break;
            case GT: r = x > y; break;
            case GE: r = x >= y; break;
            case EQ: r = std::fabs(x - y) <= d.p[0]; break;
            case AND: r = (x != 0.0) && (y != 0.0); break;
            default: r = (x != 0.0) || (y != 0.0); break;
          }
          o.v[k] = r;
        }
        return;
      }
      case POW: for (int k = 0; k < n; ++k) o.v[k] = std::pow(in(0).v[k], d.p[0]); return;
      case NEG: for (int k = 0; k < n; ++k) o.v[k] = -in(0).v[k]; return;
      case ABS: for (int k = 0; k < n; ++k) o.v[k] = std::fabs(in(0).v[k]); return;
      case NOT: for (int k = 0; k < n; ++k) o.v[k] = in(0).v[k] == 0.0; return;
      case INDEX: { const Val& a = in(0); int idx = d.i[0] >= 0 ? d.i[0] : a.n + d.i[0]; o.v[0] = a.v[idx]; return; }
      case NORM: { const Val& a = in(0); double sum = 0; for (int k = 0; k < a.n; ++k) sum += a.v[k] * a.v[k]; o.v[0] = std::sqrt(sum); return; }
      case DOT: { const Val &a = in(0), &b = in(1); o.v[0] = a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2]; return; }
      case CROSS: cross3(in(0).v, in(1).v, o.v); return;
      case SIN: for (int k = 0; k < n; ++k) o.v[k] = std::sin(in(0).v[k]); return;
      case COS: for (int k = 0; k < n; ++k) o.v[k] = std::cos(in(0).v[k]); return;
      case CLIP:
        for (int k = 0; k < n; ++k) {
          double x = in(0).v[k];
          if (!std::isnan(x)) {
            if (d.i[0] && x < d.p[0]) x = d.p[0];
            if (d.i[1] && x > d.p[1]) x = d.p[1];
          }
          o.v[k] = x;
        }
        return;
      case WHERE:
        for (int k = 0; k < n; ++k) o.v[k] = broadcast(in(0), k) != 0.0 ? broadcast(in(1), k) : broadcast(in(2), k);
        return;
      case MAXOF: case MINOF: case MEANOF: case SUMOF:
        for (int k = 0; k < n; ++k) {
          double acc = broadcast(in(0), k);
          for (size_t j = 1; j < d.in.size(); ++j) {
            double x = broadcast(in(j), k);
            if (d.op == MAXOF) acc = (std::isnan(acc) || std::isnan(x)) ? kNaN : std::max(acc, x);
            else if (d.op == MINOF) acc = (std::isnan(acc) || std::isnan(x)) ? kNaN : std::min(acc, x);
            else acc += x;
          }
          o.v[k] = d.op == MEANOF ? acc / static_cast<double>(d.in.size()) : acc;
        }
        return;
      case VEC: o.v[0] = in(0).v[0]; o.v[1] = in(1).v[0]; o.v[2] = in(2).v[0]; return;
      case UNIT: {
        const Val& a = in(0);
        double nr = std::sqrt(a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2]);
        for (int k = 0; k < 3; ++k) o.v[k] = nr < 1e-9 ? 0.0 : a.v[k] / nr;
        return;
      }
      case CLIPNORM: {
        const Val &x = in(0), &lim = in(1);
        double nr = std::sqrt(x.v[0] * x.v[0] + x.v[1] * x.v[1] + x.v[2] * x.v[2]);
        double limit = lim.v[0];
        if (nr <= limit || nr < 1e-12) { for (int k = 0; k < 3; ++k) o.v[k] = x.v[k]; }
        else { double f = std::max(limit, 0.0) / nr; for (int k = 0; k < 3; ++k) o.v[k] = x.v[k] * f; }
        return;
      }
      case COPY: for (int k = 0; k < n; ++k) o.v[k] = in(0).v[k]; return;
      case DERIV: {
        const Val& x = in(0);
        if (!s.has) { for (int k = 0; k < n; ++k) o.v[k] = kNaN; }
        else {
          double dt = t - s.t;
          for (int k = 0; k < n; ++k) {
            double dd = x.v[k] - s.a[k];
            if (d.i[0]) dd = wrap_angle(dd);
            o.v[k] = dt > 0 ? dd / dt : kNaN;
          }
        }
        s.t = t; s.has = true;
        for (int k = 0; k < n; ++k) s.a[k] = x.v[k];
        return;
      }
      case INTEGRAL: {
        const Val& x = in(0);
        if (!s.has) { for (int k = 0; k < n; ++k) s.b[k] = 0.0; }
        else { for (int k = 0; k < n; ++k) s.b[k] = s.b[k] + 0.5 * (x.v[k] + s.a[k]) * (t - s.t); }
        s.t = t; s.has = true;
        for (int k = 0; k < n; ++k) { s.a[k] = x.v[k]; o.v[k] = s.b[k]; }
        return;
      }
      case RUNMAX: case RUNMIN: {
        const Val& x = in(0);
        for (int k = 0; k < n; ++k) {
          if (!s.has) s.a[k] = x.v[k];
          else {
            double m = s.a[k], y = x.v[k];  // np.fmax / np.fmin: a NaN operand is ignored
            if (std::isnan(m)) m = y; else if (!std::isnan(y)) m = d.op == RUNMAX ? std::max(m, y) : std::min(m, y);
            s.a[k] = m;
          }
          o.v[k] = s.a[k];
        }
        s.has = true;
        return;
      }
      case ATRESET: case SINCERESET: {
        const Val& x = in(0);
        if (!s.has) { for (int k = 0; k < n; ++k) s.a[k] = x.v[k]; s.has = true; }
        for (int k = 0; k < n; ++k) o.v[k] = d.op == ATRESET ? s.a[k] : x.v[k] - s.a[k];
        return;
      }
      case WINDOW: {
        s.buf.emplace_back(t, in(0));
        while (s.buf.front().first < t - d.p[0] - 1e-12) s.buf.pop_front();
        const double cnt = static_cast<double>(s.buf.size());
        for (int k = 0; k < n; ++k) {
          double acc = s.buf.front().second.v[k];
          int stat = d.i[0];  // 0 mean 1 min 2 max 3 std 4 sum
          if (stat == 1 || stat == 2) {
            for (size_t j = 1; j < s.buf.size(); ++j) {
              double y = s.buf[j].second.v[k];
              acc = (std::isnan(acc) || std::isnan(y)) ? kNaN : (stat == 1 ? std::min(acc, y) : std::max(acc, y));
            }
            o.v[k] = acc;
          } else {
            for (size_t j = 1; j < s.buf.size(); ++j) acc += s.buf[j].second.v[k];
            if (stat == 4) o.v[k] = acc;
            else if (stat == 0) o.v[k] = acc / cnt;
            else {
              double mean = acc / cnt, sq = 0;
              for (size_t j = 0; j < s.buf.size(); ++j) { double e = s.buf[j].second.v[k] - mean; sq += e * e; }
              o.v[k] = std::sqrt(sq / cnt);
            }
          }
        }
        return;
      }
      case DURTRUE:
        if (in(0).v[0] != 0.0) { if (!s.has) { s.t = t; s.has = true; } o.v[0] = t - s.t; }
        else { s.has = false; o.v[0] = 0.0; }
        return;
      case TIMESINCE: {
        bool on = in(0).v[0] != 0.0;
        if (on && !s.prev) { s.last = t; s.has_last = true; }
        s.prev = on;
        o.v[0] = s.has_last ? t - s.last : kNaN;
        return;
      }
      case RISING: { bool on = in(0).v[0] != 0.0; o.v[0] = on && !s.prev; s.prev = on; return; }
      case FALLING: { bool on = in(0).v[0] != 0.0; o.v[0] = !on && s.prev; s.prev = on; return; }
      case LATCH: {
        bool held = s.flag;
        if (in(1).v[0] != 0.0) held = false; else if (in(0).v[0] != 0.0) held = true;
        s.flag = held; o.v[0] = held; return;
      }
      case HOLD: {
        bool on = in(0).v[0] != 0.0;
        if (on) { s.last = t; s.has_last = true; }
        o.v[0] = (s.has_last && t - s.last < d.p[0] - 1e-9) || on;
        return;
      }
      case COOLDOWN: {
        bool on = in(0).v[0] != 0.0;
        bool edge = on && !s.prev;
        s.prev = on;
        bool accept = edge && (!s.has_last || t - s.last >= d.p[0] - 1e-9);
        if (accept) { s.last = t; s.has_last = true; }
        o.v[0] = accept; return;
      }
      case ONCE: {
        bool on = in(0).v[0] != 0.0;
        bool edge = on && !s.prev;
        s.prev = on;
        bool fire = edge && !s.flag;
        if (fire) s.flag = true;
        o.v[0] = fire; return;
      }
      case SAMPLEON: {
        const Val& x = in(0);
        bool on = in(1).v[0] != 0.0;
        if (!s.has || (on && !s.prev)) { for (int k = 0; k < n; ++k) s.a[k] = x.v[k]; s.has = true; }
        s.prev = on;
        for (int k = 0; k < n; ++k) o.v[k] = s.a[k];
        return;
      }
      case P_POS: thing_pos(things[id][0], o.v); return;
      case P_HEIGHT: { double p[3]; thing_pos(things[id][0], p); o.v[0] = p[2]; return; }
      case P_VEL: { double w[3]; thing_twist(things[id][0], o.v, w); return; }
      case P_SPEED: { double v[3], w[3]; thing_twist(things[id][0], v, w); o.v[0] = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); return; }
      case P_ANGVEL: { double v[3]; thing_twist(things[id][0], v, o.v); return; }
      case P_SPIN: { double v[3], w[3]; thing_twist(things[id][0], v, w); o.v[0] = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); return; }
      case P_ORIENT: {
        double q[4];
        thing_quat(things[id][0], q);
        double sgn = q[0] < 0 ? -1.0 : 1.0;
        for (int k = 0; k < 4; ++k) o.v[k] = sgn * q[k];
        return;
      }
      case P_JQ: o.v[0] = jq[d.i[0]]; return;
      case P_JQD: o.v[0] = jqd[d.i[0]]; return;
      case P_JEFF: o.v[0] = jeff[d.i[0]]; return;
      case P_EVENT: o.v[0] = events[d.i[0]] != 0; return;
      case P_BSTATE: o.v[0] = static_cast<double>(behs[d.i[0]].state_idx); return;
      case P_BINSTATE: o.v[0] = behs[d.i[0]].state_idx == d.i[1]; return;
      case P_BTIME: o.v[0] = std::max(0.0, t - behs[d.i[0]].enter_t); return;
      case P_BVAR: { const Val& vv = behs[d.i[0]].vars[d.i[1]]; for (int k = 0; k < n; ++k) o.v[k] = vv.v[k]; return; }
      case P_FIMPULSE: o.v[0] = laws[d.i[0]].impulse; return;
      case P_FACTIVETIME: o.v[0] = laws[d.i[0]].active_time; return;
      case P_FACTIVE: o.v[0] = laws[d.i[0]].active; return;
      case P_FAPPLIED: for (int k = 0; k < 3; ++k) o.v[k] = laws[d.i[0]].last[k]; return;
      case REL_DIST: {
        double a[3], b[3];
        thing_pos(things[id][0], a);
        thing_pos(things[id][1], b);
        double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        o.v[0] = std::sqrt(dx * dx + dy * dy + dz * dz);
        return;
      }
      default: throw std::runtime_error("unknown native opcode");
    }
  }
};

}  // namespace lobs
