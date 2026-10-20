// LoBS native tick kernel (MuJoCo): physics step + state read, a measure interpreter, and a runtime for
// force laws and state-machine behaviors. See docs/performance.md.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernel.hpp"
#include "program.hpp"
#include "runtime.hpp"

namespace nb = nanobind;
using f64 = nb::ndarray<const double, nb::c_contig>;
using Out = nb::ndarray<nb::numpy, double>;
using lobs::Frame;

static Out make_array(std::vector<double> v, std::initializer_list<size_t> shape) {
  auto* heap = new std::vector<double>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  std::vector<size_t> shp(shape);
  return Out(heap->data(), shp.size(), shp.data(), owner);
}

static nb::tuple frame_tuple(const Frame& f, size_t D, size_t ndof) {
  return nb::make_tuple(f.time, make_array(f.pos, {D, 3}), make_array(f.quat, {D, 4}), make_array(f.lin, {D, 3}),
                        make_array(f.ang, {D, 3}), make_array(f.q, {ndof}), make_array(f.qd, {ndof}),
                        make_array(f.eff, {ndof}));
}

static void step_raw(uintptr_t model, uintptr_t data, int n) {
  if (!lobs::mj.ok) throw std::runtime_error("call init_mujoco() first");
  auto* m = reinterpret_cast<mjModel*>(model);
  auto* d = reinterpret_cast<mjData*>(data);
  for (int i = 0; i < n; ++i) lobs::mj.step(m, d);
}

static std::vector<lobs::WrenchRow> rows_from(f64 w) {
  std::vector<lobs::WrenchRow> rows;
  if (w.ndim() != 2) return rows;
  const double* p = w.data();
  for (size_t k = 0; k < w.shape(0); ++k, p += 10) {
    lobs::WrenchRow r;
    r.body = static_cast<int>(p[0]);
    for (int c = 0; c < 3; ++c) {
      r.f[c] = p[1 + c];
      r.tau[c] = p[4 + c];
      r.pt[c] = p[7 + c];
    }
    rows.push_back(r);
  }
  return rows;
}

using Th = std::tuple<int, int, bool>;
static lobs::Thing thing_of(const Th& t) { return lobs::Thing{std::get<0>(t), std::get<1>(t), std::get<2>(t)}; }

NB_MODULE(_native, mod) {
  mod.doc() = "LoBS native tick kernel";
  mod.attr("HEADER_VERSION") = mjVERSION_HEADER;
  mod.def("init_mujoco", &lobs::init_mujoco);
  mod.def("step_raw", &step_raw);

  nb::class_<lobs::Kernel>(mod, "Kernel")
      .def(nb::init<uintptr_t, uintptr_t, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>,
                    std::vector<int>, int>())
      .def("set_targets", [](lobs::Kernel& k, f64 u) { k.set_targets(u.data(), u.size()); })
      .def("get_targets", [](lobs::Kernel& k) { return make_array(k.targets, {k.targets.size()}); })
      .def("set_effort", [](lobs::Kernel& k, f64 e) { k.set_effort(e.data(), e.size()); })
      .def("step", [](lobs::Kernel& k, int substeps, f64 w) { k.step_rows(substeps, rows_from(w)); })
      .def("read", [](lobs::Kernel& k) {
        Frame f;
        k.read_raw(f);
        return frame_tuple(f, k.ids.size(), static_cast<size_t>(k.n_dof));
      });

  using lobs::Program;
  nb::class_<Program>(mod, "Program")
      .def(nb::init<>())
      .def("init", &Program::init)
      .def("set_node", [](Program& P, int i, int op, int kind, int n, int ndim, std::vector<int> in,
                          std::vector<double> p, std::vector<int> ints, bool eager) {
             lobs::NodeDesc& d = P.nodes[i];
             d.op = op; d.kind = kind; d.n = n; d.ndim = ndim; d.in = std::move(in); d.eager = eager;
             for (size_t k = 0; k < p.size() && k < 4; ++k) d.p[k] = p[k];
             for (size_t k = 0; k < ints.size() && k < 4; ++k) d.i[k] = ints[k];
           })
      .def("set_things", [](Program& P, int i, std::vector<Th> ths) {
             P.things[i].clear();
             for (auto& t : ths) P.things[i].push_back(thing_of(t));
           })
      .def("add_body", [](Program& P, bool is_static, int dyn, std::vector<double> wp, std::vector<double> wq,
                          std::vector<double> com, double mass) {
             lobs::BodyInfo b; b.is_static = is_static; b.dyn = dyn; b.mass = mass;
             for (int k = 0; k < 3; ++k) { b.wp[k] = wp[k]; b.com[k] = com[k]; }
             for (int k = 0; k < 4; ++k) b.wq[k] = wq[k];
             P.bodies.push_back(b);
           })
      .def("add_actor", [](Program& P, std::vector<int> bodies, double mass) {
             lobs::ActorInfo a; a.bodies = std::move(bodies); a.mass = mass; P.actors.push_back(std::move(a));
           })
      .def("add_anchor", [](Program& P, int parent, std::vector<double> pos, std::vector<double> quat, int rule) {
             lobs::AnchorInfo a; a.parent = parent; a.rule = rule;
             for (int k = 0; k < 3; ++k) a.pos[k] = pos[k];
             for (int k = 0; k < 4; ++k) a.quat[k] = quat[k];
             P.anchors.push_back(a);
           })
      .def("set_order", [](Program& P, std::vector<int> order) { P.order = std::move(order); })
      .def("set_event_count", [](Program& P, int n) { P.events.assign(n, 0); })
      .def("finalize", &Program::finalize)
      .def("reset", &Program::reset)
      .def("begin_tick", &Program::begin_tick)
      .def("eval_all", &Program::eval_all);

  using lobs::Runtime;
  nb::class_<Runtime>(mod, "Runtime")
      .def(nb::init<>())
      .def("configure", [](Runtime& R, Program& P, lobs::Kernel& K, double tick_dt, int substeps,
                           std::vector<int> mj_body, int n_events, nb::object err_cls) {
             R.prog = &P; R.kernel = &K; R.tick_dt = tick_dt; R.substeps = substeps; R.mj_body = std::move(mj_body);
             R.err_cls = err_cls; R.events_next.assign(n_events, 0); P.events.assign(n_events, 0);
           })
      .def("add_law", [](Runtime& R, std::string name, Th on, bool has_toward, Th toward, bool has_dir,
                         std::vector<double> dir, bool has_axis, Th axis_thing, std::vector<double> axis_local,
                         bool reaction, bool has_vector, bool has_when, int node_mag, int node_vec, int node_when,
                         bool has_peak, double peak, bool has_imp, double impulse, bool has_act, double active_time) {
             lobs::LawDef L;
             L.name = std::move(name); L.on = thing_of(on); L.has_toward = has_toward; L.toward = thing_of(toward);
             L.has_dir = has_dir; L.has_axis = has_axis; L.axis_thing = thing_of(axis_thing); L.reaction = reaction;
             L.has_vector = has_vector; L.has_when = has_when; L.node_mag = node_mag; L.node_vec = node_vec;
             L.node_when = node_when; L.has_peak = has_peak; L.peak = peak; L.has_imp = has_imp; L.impulse = impulse;
             L.has_act = has_act; L.active_time = active_time;
             for (int k = 0; k < 3 && k < (int)dir.size(); ++k) L.dir[k] = dir[k];
             for (int k = 0; k < 3 && k < (int)axis_local.size(); ++k) L.axis_local[k] = axis_local[k];
             R.laws.push_back(std::move(L));
           })
      .def("add_behavior", [](Runtime& R, std::string name, bool has_on, Th on, int period, int latency, int initial,
                              bool enabled0, std::vector<int> var_n, std::vector<double> var_init, int n_states) {
             lobs::BehDef b;
             b.name = std::move(name); b.has_on = has_on; b.on = thing_of(on); b.period = period; b.latency = latency;
             b.initial = initial; b.enabled0 = enabled0; b.var_n = var_n; b.states.resize(n_states);
             size_t off = 0;
             for (int n : var_n) {
               lobs::Val v; v.n = n;
               for (int k = 0; k < n; ++k) v.v[k] = var_init[off + k];
               off += n;
               b.var_init.push_back(v);
             }
             R.behs.push_back(std::move(b));
           })
      .def("add_action", [](Runtime& R, int beh, int state, int phase, int type, int act_idx, int node, bool has_on,
                            Th on, bool has_max, double max_force, int mode, bool has_target, Th target, double speed,
                            double max_accel, double gain, double stop_radius, bool plane, int var, int event, int idx,
                            bool flag) {
             lobs::Act a;
             a.type = type; a.act_idx = act_idx; a.node = node; a.has_on = has_on; a.on = thing_of(on);
             a.has_max = has_max; a.max_force = max_force; a.mode = mode; a.has_target = has_target;
             a.target = thing_of(target); a.speed = speed; a.max_accel = max_accel; a.gain = gain;
             a.stop_radius = stop_radius; a.plane = plane; a.var = var; a.event = event; a.idx = idx; a.flag = flag;
             auto& st = R.behs[beh].states[state];
             (phase == 0 ? st.enter : phase == 1 ? st.run : st.exit).push_back(a);
           })
      .def("add_transition", [](Runtime& R, int beh, int from, int to, int node, double min_time, int priority) {
             lobs::Trans t; t.from = from; t.to = to; t.node = node; t.min_time = min_time; t.priority = priority;
             R.behs[beh].trans.push_back(t);
           })
      .def("finalize", &Runtime::finalize)
      .def("reset", &Runtime::reset)
      .def("observe", [](Runtime& R, f64 pos, f64 quat, f64 lin, f64 ang, f64 q, f64 qd, f64 eff, double t) {
             Frame f;
             f.time = t;
             f.pos.assign(pos.data(), pos.data() + pos.size());
             f.quat.assign(quat.data(), quat.data() + quat.size());
             f.lin.assign(lin.data(), lin.data() + lin.size());
             f.ang.assign(ang.data(), ang.data() + ang.size());
             f.q.assign(q.data(), q.data() + q.size());
             f.qd.assign(qd.data(), qd.data() + qd.size());
             f.eff.assign(eff.data(), eff.data() + eff.size());
             R.observe(f, t);
             return R.prog->values_array();
           })
      .def("step", [](Runtime& R, f64 user_rows) {
             R.step_tick(rows_from(user_rows));
             const size_t D = R.cur.pos.size() / 3, nd = R.cur.q.size();
             return nb::make_tuple(frame_tuple(R.cur, D, nd), R.prog->values_array());
           })
      .def("behavior_view", [](Runtime& R, int i) {
             const lobs::BState& s = R.prog->behs[i];
             std::vector<double> flat;
             for (auto& v : s.vars) flat.insert(flat.end(), v.v, v.v + v.n);
             return nb::make_tuple(s.state_idx, s.enter_t, s.enabled, flat);
           })
      .def("law_view", [](Runtime& R, int i) {
             const lobs::LawState& s = R.prog->laws[i];
             return nb::make_tuple(s.impulse, s.active_time, s.peak, s.active,
                                   std::vector<double>{s.last[0], s.last[1], s.last[2]}, s.enabled);
           })
      .def("set_law_enabled", [](Runtime& R, int i, bool f) { R.prog->laws[i].enabled = f; })
      .def("set_behavior_enabled", [](Runtime& R, int i, bool f) { R.prog->behs[i].enabled = f; })
      .def("get_log", [](Runtime& R) {
             std::vector<std::tuple<double, int, int, int>> out;
             for (auto& e : R.log) out.emplace_back(e.t, e.beh, e.from, e.to);
             return out;
           })
      .def("events", [](Runtime& R) {
             std::vector<int> out;
             for (size_t i = 0; i < R.prog->events.size(); ++i) if (R.prog->events[i]) out.push_back((int)i);
             return out;
           })
      .def_prop_ro("tick", [](Runtime& R) { return R.tick; });
}
