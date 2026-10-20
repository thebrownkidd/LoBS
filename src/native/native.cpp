// LoBS native tick kernel. B0: build smoke test + MuJoCo access through the library the Python package loaded.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include <stdexcept>
#include <mujoco/mujoco.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nb = nanobind;

struct MJ {
  decltype(&mj_step) step = nullptr;
  decltype(&mj_versionString) version = nullptr;
  bool ok = false;
};
static MJ mj;

static void* sym(const char* name) {
#ifdef _WIN32
  HMODULE h = GetModuleHandleA("mujoco.dll");
  if (!h) throw std::runtime_error("mujoco.dll is not loaded; import mujoco first");
  return reinterpret_cast<void*>(GetProcAddress(h, name));
#else
  return dlsym(RTLD_DEFAULT, name);
#endif
}

static std::string init_mujoco() {
  mj.step = reinterpret_cast<decltype(&mj_step)>(sym("mj_step"));
  mj.version = reinterpret_cast<decltype(&mj_versionString)>(sym("mj_versionString"));
  if (!mj.step || !mj.version) throw std::runtime_error("cannot resolve MuJoCo symbols");
  std::string loaded = mj.version();
  std::string built = std::to_string(mjVERSION_HEADER);
  mj.ok = true;
  return loaded + " (headers " + built + ")";
}

static void step_raw(uintptr_t model, uintptr_t data, int n) {
  if (!mj.ok) throw std::runtime_error("call init_mujoco() first");
  auto* m = reinterpret_cast<mjModel*>(model);
  auto* d = reinterpret_cast<mjData*>(data);
  for (int i = 0; i < n; ++i) mj.step(m, d);
}

NB_MODULE(_native, m) {
  m.doc() = "LoBS native tick kernel";
  m.attr("HEADER_VERSION") = mjVERSION_HEADER;
  m.def("init_mujoco", &init_mujoco);
  m.def("step_raw", &step_raw);
}
