// Access to MuJoCo through the library the Python package already loaded (one copy of the engine per process).
#pragma once
#include <stdexcept>
#include <string>
#include <mujoco/mujoco.h>
#ifdef _WIN32
// kernel32 only; windows.h is avoided because its macros (CONST, ...) collide with our opcode names
extern "C" __declspec(dllimport) void* __stdcall GetModuleHandleA(const char*);
extern "C" __declspec(dllimport) void* __stdcall GetProcAddress(void*, const char*);
#else
#include <dlfcn.h>
#endif

namespace lobs {

struct MJ {
  decltype(&mj_step) step = nullptr;
  decltype(&mj_versionString) version = nullptr;
  decltype(&mj_applyFT) applyFT = nullptr;
  decltype(&mj_kinematics) kinematics = nullptr;
  decltype(&mj_comPos) comPos = nullptr;
  decltype(&mj_comVel) comVel = nullptr;
  bool ok = false;
};
inline MJ mj;

inline void* sym(const char* name) {
#ifdef _WIN32
  void* h = GetModuleHandleA("mujoco.dll");
  if (!h) throw std::runtime_error("mujoco.dll is not loaded; import mujoco first");
  void* p = GetProcAddress(h, name);
#else
  void* p = dlsym(RTLD_DEFAULT, name);
#endif
  if (!p) throw std::runtime_error(std::string("cannot resolve MuJoCo symbol ") + name);
  return p;
}

template <class F>
inline F load(const char* name) {
  return reinterpret_cast<F>(sym(name));
}

inline std::string init_mujoco() {
  mj.step = load<decltype(&mj_step)>("mj_step");
  mj.version = load<decltype(&mj_versionString)>("mj_versionString");
  mj.applyFT = load<decltype(&mj_applyFT)>("mj_applyFT");
  mj.kinematics = load<decltype(&mj_kinematics)>("mj_kinematics");
  mj.comPos = load<decltype(&mj_comPos)>("mj_comPos");
  mj.comVel = load<decltype(&mj_comVel)>("mj_comVel");
  mj.ok = true;
  return std::string(mj.version()) + " (headers " + std::to_string(mjVERSION_HEADER) + ")";
}

}  // namespace lobs
