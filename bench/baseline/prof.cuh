// Stage timers for timing the original code (bench/baseline; not part of
// the tag baseline-2026-09). Wall-clock (steady_clock) per stage + NVTX
// ranges so nsys timelines line up with the table. Usage:
//   StageTimer st("obj0/");  st.begin("read_csr"); ... st.begin("build_adj"); ... st.end();
//   profCounter("obj0/iterations") += 1;
#pragma once
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#if defined(__CUDACC__) || defined(PROF_USE_NVTX)
#include <nvtx3/nvToolsExt.h>
#define PROF_NVTX 1
#endif

struct ProfRec {
  std::string name;
  double ms;
};
inline std::vector<ProfRec> &profRecs() {
  static std::vector<ProfRec> r;
  return r;
}
inline std::map<std::string, double> &profCounters() {
  static std::map<std::string, double> c;
  return c;
}
inline double &profCounter(const std::string &k) { return profCounters()[k]; }
inline std::string &profPrefix() {
  static std::string p;
  return p;
}

class StageTimer {
public:
  explicit StageTimer(const std::string &prefix) : prefix_(prefix) {}
  ~StageTimer() { end(); }
  void begin(const std::string &stage) {
    end();
    cur_ = prefix_ + stage;
    open_ = true;
#ifdef PROF_NVTX
    nvtxRangePushA(cur_.c_str());
#endif
    t0_ = std::chrono::steady_clock::now();
  }
  void end() {
    if (!open_) return;
    auto t1 = std::chrono::steady_clock::now();
#ifdef PROF_NVTX
    nvtxRangePop();
#endif
    profRecs().push_back(
        {cur_, std::chrono::duration<double, std::milli>(t1 - t0_).count()});
    open_ = false;
  }

private:
  std::string prefix_, cur_;
  bool open_ = false;
  std::chrono::steady_clock::time_point t0_;
};

inline void profDump(FILE *f = stdout) {
  std::fprintf(f, "\n==== STAGE TIMES (wall ms) ====\n");
  double tot = 0;
  for (auto &r : profRecs()) {
    std::fprintf(f, "STAGE %-55s %12.3f\n", r.name.c_str(), r.ms);
  }
  std::fprintf(f, "==== COUNTERS ====\n");
  for (auto &kv : profCounters())
    std::fprintf(f, "COUNTER %-53s %14.0f\n", kv.first.c_str(), kv.second);
  (void)tot;
}
