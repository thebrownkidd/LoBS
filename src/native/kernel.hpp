// Native physics step + state read for the MuJoCo backend.
#pragma once
#include <algorithm>
#include <cstring>
#include <vector>
#include "mjapi.hpp"

namespace lobs {

struct Frame {
  double time = 0;
  std::vector<double> pos, quat, lin, ang, q, qd, eff;
};

struct WrenchRow {
  int body;  // MuJoCo body id
  double f[3], tau[3], pt[3];
};

class Kernel {
 public:
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

  void set_targets(const double* u, size_t n) {
    n = std::min<size_t>(n, targets.size());
    for (size_t i = 0; i < n; ++i) targets[i] = u[i];
  }

  void set_effort(const double* e, size_t n) {
    n = std::min<size_t>(n, effort_applied.size());
    for (size_t i = 0; i < n; ++i) effort_applied[i] = e[i];
  }

  // Wrench rows on the same body and point are summed in order (one native call instead of one each).
  void step_rows(int substeps, const std::vector<WrenchRow>& rows) {
    if (m->nu) std::memcpy(d->ctrl, targets.data(), sizeof(double) * m->nu);
    if (rows.empty()) {
      std::memset(d->qfrc_applied, 0, sizeof(double) * m->nv);
      for (int i = 0; i < substeps; ++i) mj.step(m, d);
    } else {
      std::vector<WrenchRow> merged;
      merged.reserve(rows.size());
      for (const auto& w : rows) {
        WrenchRow* hit = nullptr;
        for (auto& x : merged) {
          if (x.body == w.body && std::memcmp(x.pt, w.pt, 3 * sizeof(double)) == 0) {
            hit = &x;
            break;
          }
        }
        if (!hit) {
          merged.push_back(w);
        } else {
          for (int c = 0; c < 3; ++c) {
            hit->f[c] += w.f[c];
            hit->tau[c] += w.tau[c];
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

  void read_raw(Frame& out) {
    mj.kinematics(m, d);
    mj.comPos(m, d);
    mj.comVel(m, d);
    const size_t D = ids.size();
    out.pos.assign(3 * D, 0.0);
    out.quat.assign(4 * D, 0.0);
    out.lin.assign(3 * D, 0.0);
    out.ang.assign(3 * D, 0.0);
    for (size_t k = 0; k < D; ++k) {
      const int b = ids[k];
      const double* xp = d->xpos + 3 * b;
      const double* xq = d->xquat + 4 * b;
      const double* cv = d->cvel + 6 * b;
      const double* sc = d->subtree_com + 3 * rootid[k];
      for (int c = 0; c < 3; ++c) out.pos[3 * k + c] = xp[c];
      for (int c = 0; c < 4; ++c) out.quat[4 * k + c] = xq[c];
      const double a0 = cv[0], a1 = cv[1], a2 = cv[2];
      const double r0 = xp[0] - sc[0], r1 = xp[1] - sc[1], r2 = xp[2] - sc[2];
      out.ang[3 * k + 0] = a0;
      out.ang[3 * k + 1] = a1;
      out.ang[3 * k + 2] = a2;
      out.lin[3 * k + 0] = cv[3] + (a1 * r2 - a2 * r1);
      out.lin[3 * k + 1] = cv[4] + (a2 * r0 - a0 * r2);
      out.lin[3 * k + 2] = cv[5] + (a0 * r1 - a1 * r0);
    }
    out.q.assign(n_dof, 0.0);
    out.qd.assign(n_dof, 0.0);
    out.eff.assign(n_dof, 0.0);
    for (size_t i = 0; i < s_dof.size(); ++i) {
      out.q[s_dof[i]] = d->qpos[s_qa[i]];
      out.qd[s_dof[i]] = d->qvel[s_da[i]];
    }
    for (size_t a = 0; a < act_dof.size(); ++a) out.eff[act_dof[a]] += effort_applied[a];
    out.time = static_cast<double>(d->time);
  }
};

}  // namespace lobs
