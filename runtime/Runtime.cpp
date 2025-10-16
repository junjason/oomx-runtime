#include "oomx/Runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace oomx {

using clk = std::chrono::high_resolution_clock;
Runtime::RunStats Runtime::g_stats_{};

Runtime::Runtime(Store* s): store_(s) {}
Runtime::~Runtime() = default;

void Runtime::ingest(const Sow&) { /* external events not used in MVP */ }

Snapshot Runtime::snapshot() const { return {}; }

// One-time active-first index build (zero-copy “reorder”)
void Runtime::maybeBuildIndexOnce() {
  if (!dynamic_ || !use_idx_ || dyn_done_) return;

  const uint32_t N = store_->size();
  if (!N) { dyn_done_ = true; return; }

  uint32_t* state = nullptr;
  try { state = store_->column<uint32_t>("state"); } catch (...) { state = nullptr; }
  if (!state) { dyn_done_ = true; return; }

  // Init idx if first use
  if (idx_.size() != N) {
    idx_.resize(N);
    for (uint32_t i=0;i<N;++i) idx_[i] = i;
  }

  // Check mix ratio to decide if grouping helps
  uint64_t active=0; for (uint32_t i=0;i<N;++i) active += (state[i]==1u);
  double ratio = double(active) / double(N);
  if (ratio <= dyn_low_ || ratio >= dyn_high_) { dyn_done_ = true; return; }

  // Build ACTIVE-first layout in idx_ (no data movement)
  std::vector<uint32_t> activeRows; activeRows.reserve(N);
  std::vector<uint32_t> otherRows;  otherRows.reserve(N);
  for (uint32_t i=0;i<N;++i) ((state[i]==1u) ? activeRows : otherRows).push_back(i);

  uint32_t k=0;
  for (auto r: activeRows) idx_[k++] = r;
  for (auto r: otherRows)  idx_[k++] = r;

  dyn_done_ = true; // one-shot; extend later if you want periodic updates
}

void Runtime::run(Pipeline& pipe) {
  auto t0 = clk::now();

  const uint32_t N    = store_->size();
  const uint32_t TILE = tile_rows_;

  // init per-frame Sow only when deterministic
  frameSow_.reset(deterministic_ ? new Sow() : nullptr);
  Sow* sowPtr = frameSow_.get();

  // Initialize idx on first use; possibly build ACTIVE-first once
  if (use_idx_ && idx_.size() != N) {
    idx_.resize(N);
    for (uint32_t i=0;i<N;++i) idx_[i] = i;
  }
  maybeBuildIndexOnce(); // no-op if conditions not met

  // Run kernels over tiles
  for (auto& kv : pipe.maps()) {
    auto& fn = kv.second;
    for (uint32_t b = 0; b < N; b += TILE) {
      KernelCtx k{ snapshot(), sowPtr, b, std::min<uint32_t>(b+TILE, N),
                   (use_idx_ && !idx_.empty()) ? idx_.data() : nullptr };
      fn(k);
    }
  }

  auto t1 = clk::now();
  double us = std::chrono::duration<double,std::micro>(t1 - t0).count();
  g_stats_.samples.push_back(us);
  ++tick_;
}

void Runtime::resolveAndCommit() {
  if (!frameSow_) return; // in-place mode
  const uint32_t* idxPtr = (use_idx_ && !idx_.empty()) ? idx_.data() : nullptr;

  // For each field present in this Sow, apply set then add (stable row order)
  for (auto& fname : frameSow_->fieldNamesF32()) {
    auto& fb = frameSow_->fieldF32(fname);

    // SETS
    if (!fb.set.empty()) {
      std::vector<std::pair<uint32_t,float>> items(fb.set.begin(), fb.set.end());
      std::sort(items.begin(), items.end(), [](auto& a, auto& b){ return a.first < b.first; });
      float* col = nullptr; try { col = store_->column<float>(fname.c_str()); } catch (...) { col = nullptr; }
      if (col) {
        for (auto& [rowLogical, val] : items) {
          uint32_t row = idxPtr ? idxPtr[rowLogical] : rowLogical;
          col[row] = val;
        }
      }
    }

    // ADDS
    if (!fb.add.empty()) {
      std::vector<std::pair<uint32_t,float>> items(fb.add.begin(), fb.add.end());
      std::sort(items.begin(), items.end(), [](auto& a, auto& b){ return a.first < b.first; });
      float* col = nullptr; try { col = store_->column<float>(fname.c_str()); } catch (...) { col = nullptr; }
      if (col) {
        for (auto& [rowLogical, delta] : items) {
          uint32_t row = idxPtr ? idxPtr[rowLogical] : rowLogical;
          col[row] += delta;
        }
      }
    }
  }
}

Metrics Runtime::metrics() const {
  Metrics m{};
  const auto& s = g_stats_.samples;
  if (s.empty()) return m;
  double sum=0.0; for (double x : s) sum += x; m.mean_us = sum / s.size();

  auto v = s; std::sort(v.begin(), v.end());
  auto pct = [&](double p){ size_t i = (size_t)((p/100.0)*(v.size()-1)); return v[i]; };
  m.p95_us = pct(95.0); m.p99_us = pct(99.0);

  double ss=0.0; for (double x : s){ double d=x-m.mean_us; ss+=d*d; }
  m.sigma_us = std::sqrt(ss / (s.size() > 1 ? (s.size()-1) : 1));
  return m;
}

} // namespace oomx
