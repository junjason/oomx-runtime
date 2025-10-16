#pragma once
#include "Schema.h"
#include "Store.h"
#include "Sow.h"
#include <functional>
#include <memory>
#include <vector>

namespace oomx {

struct Snapshot { /* opaque for MVP */ };

struct Metrics {
  double mean_us=0, p95_us=0, p99_us=0, sigma_us=0;
  double branch_eff=1, coalescing=1;
};

struct Policy { double min_branch=0.85, min_coalescing=0.90; int windows=3; };

struct KernelCtx {
  Snapshot snap;
  Sow* sow;                   // nullptr in in-place mode; valid in deterministic mode
  uint32_t tileBegin, tileEnd;
  const uint32_t* idx = nullptr; // OPTIONAL logical->physical row map (index indirection)
};

using KernelFn = std::function<void(KernelCtx&)>;

class Pipeline {
public:
  void addMap(const char* name, KernelFn fn) { maps_.push_back({name, std::move(fn)}); }
  const auto& maps() const { return maps_; }
private:
  std::vector<std::pair<std::string, KernelFn>> maps_;
};

class Runtime {
public:
  explicit Runtime(Store* s);
  ~Runtime();

  void ingest(const Sow& external);
  Snapshot snapshot() const;

  void run(Pipeline&);              // executes kernels over tiles
  void resolveAndCommit();          // applies Sow (deterministic writes)
  Metrics metrics() const;

  void setPolicy(Policy p) { policy_ = p; }
  void setTileRows(uint32_t r) { tile_rows_ = r ? r : tile_rows_; }
  void setDeterministic(bool d) { deterministic_ = d; }
  void setDynamic(bool d) { dynamic_ = d; }
  void enableIndexMode(bool on) { use_idx_ = on; }   // zero-copy “reorder”

private:
  // Heuristics / helpers (MVP)
  void maybeBuildIndexOnce();       // one-time active-first index if helpful

  Store* store_;
  Policy policy_{};
  uint32_t tile_rows_ = 8192;

  bool deterministic_ = false;      // Sow + ordered commit
  bool dynamic_ = false;            // allow dynamic regrouping
  bool use_idx_ = false;            // use idx_ indirection instead of copying columns

  // simple metrics bucket
  struct RunStats { std::vector<double> samples; };
  static RunStats g_stats_;

  // dynamic controller
  uint32_t tick_ = 0;
  bool dyn_done_ = false;           // guard: build index once (MVP)
  uint32_t last_repart_tick_ = 0;
  uint32_t dyn_cooldown_ticks_ = 60;
  double   dyn_low_ = 0.30, dyn_high_ = 0.70;

  // index indirection
  std::vector<uint32_t> idx_;       // logical->physical row mapping

  // per-frame Sow (only if deterministic_)
  std::unique_ptr<Sow> frameSow_;
};

} // namespace oomx
