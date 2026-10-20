// LoBS native tick kernel (MuJoCo). Calls MuJoCo through the library the Python package already loaded,
// so there is exactly one copy of the engine in the process.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>
#include "program.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nb = nanobind;
using f64 = nb::ndarray<const double, nb::c_contig>;

// ------------------------------------------------------------------ MuJoCo function table
struct MJ {
  decltype(&mj_step) step = nullptr;
  decltype(&mj_versionString) version = nullptr;
  decltype(&mj_applyFT) applyFT = nullptr;
  decltype(&mj_kinematics) kinematics = nullptr;
  decltype(&mj_comPos) comPos = nullptr;
  decltype(&mj_comVel) comVel = nullptr;
  bool ok = false;
};
static MJ mj;

static void* sym(const char* name) {
#ifdef _WIN32
  HMODULE h = GetModuleHandleA("mujoco.dll");
  if (!h) throw std::runtime_error("mujoco.dll is not loaded; import mujoco first");
  void* p = reinterpret_cast<void*>(GetProcAddress(h, name));
#else
  void* p = dlsym(RTLD_DEFAULT, name);
#endif
  if (!p) throw std::runtime_error(std::string("cannot resolve MuJoCo symbol ") + name);
  return p;
}

template <class F>
static F load(const char* name) {
  return reinterpret_cast<F>(sym(name));
}

static std::string init_mujoco() {
  mj.step = load<decltype(&mj_step)>("mj_step");
  mj.version = load<decltype(&mj_versionString)>("mj_versionString");
  mj.applyFT = load<decltype(&mj_applyFT)>("mj_applyFT");
  mj.kinematics = load<decltype(&mj_kinematics)>("mj_kinematics");
  mj.comPos = load<decltype(&mj_comPos)>("mj_comPos");
  mj.comVel = load<decltype(&mj_comVel)>("mj_comVel");
  mj.ok = true;
  return std::string(mj.version()) + " (headers " + std::to_string(mjVERSION_HEADER) + ")";
}

static void step_raw(uintptr_t model, uintptr_t data, int n) {
  if (!mj.ok) throw std::runtime_error("call init_mujoco() first");
  auto* m = reinterpret_cast<mjModel*>(model);
  auto* d = reinterpret_cast<mjData*>(data);
  for (int i = 0; i < n; ++i) mj.step(m, d);
}

// ------------------------------------------------------------------ array helpers
using Out = nb::ndarray<nb::numpy, double>;

static Out make_array(std::vector<double>&& v, std::initializer_list<size_t> shape) {
  auto* heap = new std::vector<double>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  std::vector<size_t> shp(shape);
  return Out(heap->data(), shp.size(), shp.data(), owner);
}

// ------------------------------------------------------------------ Kernel: step + state read
struct Kernel {
  mjModel* m;
  mjData* d;
  std::vector<int> ids, rootid, s_dof, s_qa, s_da, act_dof;
  int n_dof;
  std::vector<double> targets, effort_applied;

  Kernel(uintptr_t model, uintptr_t data, std::vector<int> dyn_ids, std::vector<int> simple_dof,
         std::vector<int> simple_qa, std::vector<int> simple_da, std::vector<int> actuator_dof, int ndof)
      : m(reinterpret_cast<mjModel*>(model)),
        d(reinterpret_cast<mjData*>(data)),
        ids(std::move(dyn_ids)),
        s_dof(std::move(simple_dof)),
        s_qa(std::move(simple_qa)),
        s_da(std::move(simple_da)),
        act_dof(std::move(actuator_dof)),
        n_dof(ndof) {
    if (!mj.ok) throw std::runtime_error("call init_mujoco() first");
    for (int id : ids) rootid.push_back(m->body_rootid[id]);
    targets.assign(m->nu, 0.0);
    effort_applied.assign(m->nu, 0.0);
  }

  void set_targets(f64 u) {
    size_t n = std::min<size_t>(u.size(), targets.size());
    for (size_t i = 0; i < n; ++i) targets[i] = u.data()[i];
  }

  void set_effort(f64 e) {
    size_t n = std::min<size_t>(e.size(), effort_applied.size());
    for (size_t i = 0; i < n; ++i) effort_applied[i] = e.data()[i];
  }

  // wrenches: (K, 10) rows of [mj_body, f(3), tau(3), point(3)]; rows on the same body and point are summed in order
  void step(int substeps, f64 wrenches) {
    if (m->nu) std::memcpy(d->ctrl, targets.data(), sizeof(double) * m->nu);
    const size_t K = wrenches.ndim() == 2 ? wrenches.shape(0) : 0;
    if (K == 0) {
      std::memset(d->qfrc_applied, 0, sizeof(double) * m->nv);
      for (int i = 0; i < substeps; ++i) mj.step(m, d);
    } else {
      struct W {
        int body;
        double f[3], tau[3], pt[3];
      };
      std::vector<W> merged;
      const double* w = wrenches.data();
      for (size_t k = 0; k < K; ++k, w += 10) {
        int body = static_cast<int>(w[0]);
        W* hit = nullptr;
        for (auto& x : merged) {
          if (x.body == body && std::memcmp(x.pt, w + 7, 3 * sizeof(double)) == 0) {
            hit = &x;
            break;
          }
        }
        if (!hit) {
          merged.push_back(W{body, {w[1], w[2], w[3]}, {w[4], w[5], w[6]}, {w[7], w[8], w[9]}});
        } else {
          for (int c = 0; c < 3; ++c) {
            hit->f[c] += w[1 + c];
            hit->tau[c] += w[4 + c];
          }
        }
      }
      for (int i = 0; i < substeps; ++i) {
        std::memset(d->qfrc_applied, 0, sizeof(double) * m->nv);
        for (auto& x : merged) mj.applyFT(m, d, x.f, x.tau, x.pt, x.body, d->qfrc_applied);
        mj.step(m, d);
      }
    }
    if (m->nu) std::memcpy(effort_applied.data(), d->actuator_force, sizeof(double) * m->nu);
  }

  // (time, pos(D,3), quat(D,4), lin(D,3), ang(D,3), q, qd, effort); no contacts or accelerations
  nb::tuple read() {
    mj.kinematics(m, d);
    mj.comPos(m, d);
    mj.comVel(m, d);
    const size_t D = ids.size();
    std::vector<double> pos(3 * D), quat(4 * D), lin(3 * D), ang(3 * D);
    for (size_t k = 0; k < D; ++k) {
      const int b = ids[k];
      const double* xp = d->xpos + 3 * b;
      const double* xq = d->xquat + 4 * b;
      const double* cv = d->cvel + 6 * b;
      const double* sc = d->subtree_com + 3 * rootid[k];
      for (int c = 0; c < 3; ++c) pos[3 * k + c] = xp[c];
      for (int c = 0; c < 4; ++c) quat[4 * k + c] = xq[c];
      const double a0 = cv[0], a1 = cv[1], a2 = cv[2];
      const double r0 = xp[0] - sc[0], r1 = xp[1] - sc[1], r2 = xp[2] - sc[2];
      ang[3 * k + 0] = a0;
      ang[3 * k + 1] = a1;
      ang[3 * k + 2] = a2;
      lin[3 * k + 0] = cv[3] + (a1 * r2 - a2 * r1);
      lin[3 * k + 1] = cv[4] + (a2 * r0 - a0 * r2);
      lin[3 * k + 2] = cv[5] + (a0 * r1 - a1 * r0);
    }
    std::vector<double> q(n_dof, 0.0), qd(n_dof, 0.0), effort(n_dof, 0.0);
    for (size_t i = 0; i < s_dof.size(); ++i) {
      q[s_dof[i]] = d->qpos[s_qa[i]];
      qd[s_dof[i]] = d->qvel[s_da[i]];
    }
    for (size_t a = 0; a < act_dof.size(); ++a) effort[act_dof[a]] += effort_applied[a];
    const size_t nd = static_cast<size_t>(n_dof);
    return nb::make_tuple(static_cast<double>(d->time), make_array(std::move(pos), {D, 3}),
                          make_array(std::move(quat), {D, 4}), make_array(std::move(lin), {D, 3}),
                          make_array(std::move(ang), {D, 3}), make_array(std::move(q), {nd}),
                          make_array(std::move(qd), {nd}), make_array(std::move(effort), {nd}));
  }
};

NB_MODULE(_native, mod) {
  mod.doc() = "LoBS native tick kernel";
  mod.attr("HEADER_VERSION") = mjVERSION_HEADER;
  mod.def("init_mujoco", &init_mujoco);
  mod.def("step_raw", &step_raw);
  nb::class_<Kernel>(mod, "Kernel")
      .def(nb::init<uintptr_t, uintptr_t, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>,
                    std::vector<int>, int>())
      .def("set_targets", &Kernel::set_targets)
      .def("set_effort", &Kernel::set_effort)
      .def("step", &Kernel::step)
      .def("read", &Kernel::read);

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
      .def("set_things", [](Program& P, int i, std::vector<std::tuple<int, int, bool>> ths) {
             P.things[i].clear();
             for (auto& [k, idx, com] : ths) P.things[i].push_back(lobs::Thing{k, idx, com});
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
      .def("finalize", &Program::finalize)
      .def("reset", &Program::reset)
      .def("begin_tick", &Program::begin_tick)
      .def("eval_all", &Program::eval_all);
}
