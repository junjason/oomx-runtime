#include "oomx/Schema.h"
#include "oomx/Store.h"
#include "oomx/Runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace oomx;

struct Args {
  std::string sim = "gaming";   // gaming | finance
  std::string mode = "soa";     // aos | soa | aosoa (aosoa not differentiated in MVP)
  bool        dynamic = false;  // dynamic regrouping
  bool        det = false;      // deterministic Sow+commit
  uint32_t    n = 200000;       // entities/orders
  uint32_t    ticks = 600;      // measured frames/microbatches
  uint32_t    warmup = 0;       // unmeasured warmup ticks
  uint32_t    tile = 8192;      // tile rows (AoSoA-style)
  const char* csv = nullptr;    // optional CSV output path
};

static Args parse(int argc, char** argv) {
  Args a;
  for (int i=1; i<argc; ++i) {
    if (!std::strcmp(argv[i],"--sim")    && i+1<argc) a.sim    = argv[++i];
    else if (!std::strcmp(argv[i],"--mode")   && i+1<argc) a.mode   = argv[++i];
    else if (!std::strcmp(argv[i],"--dynamic"))                      a.dynamic = true;
    else if (!std::strcmp(argv[i],"--det"))                          a.det = true;
    else if (!std::strcmp(argv[i],"--n")      && i+1<argc) a.n      = std::stoul(argv[++i]);
    else if (!std::strcmp(argv[i],"--ticks")  && i+1<argc) a.ticks  = std::stoul(argv[++i]);
    else if (!std::strcmp(argv[i],"--warmup") && i+1<argc) a.warmup = std::stoul(argv[++i]);
    else if (!std::strcmp(argv[i],"--tile")   && i+1<argc) a.tile   = std::stoul(argv[++i]);
    else if (!std::strcmp(argv[i],"--csv")    && i+1<argc) a.csv    = argv[++i];
  }
  return a;
}

static void stats_print(const char* tag, const std::vector<double>& samples) {
  if (samples.empty()) { std::printf("%s: no samples\n", tag); return; }
  std::vector<double> v = samples; std::sort(v.begin(), v.end());
  double sum=0.0; for (double x : v) sum += x; double mean = sum / v.size();
  auto pct = [&](double p){ size_t i=(size_t)((p/100.0)*(v.size()-1)); return v[i]; };
  double ss=0.0; for (double x : v){ double d=x-mean; ss+=d*d; }
  double sigma = std::sqrt(ss / (v.size()>1 ? (v.size()-1) : 1));
  std::printf("%s mean=%.2fus p95=%.2fus p99=%.2fus sigma=%.2fus\n",
              tag, mean, pct(95), pct(99), sigma);
}

static void write_csv(const char* path, const std::vector<double>& samples) {
  if (!path) return;
  FILE* f = std::fopen(path, "w");
  if (!f) return;
  std::fprintf(f, "latency_us\n");
  for (double x : samples) std::fprintf(f, "%.3f\n", x);
  std::fclose(f);
}

// --- Gaming kernels (SoA/AoSoA)
static void build_gaming(Store& st, Pipeline& pipe, bool det) {
  // initialize once
  auto px = st.column<float>("pos_x");
  auto py = st.column<float>("pos_y");
  auto vx = st.column<float>("vel_x");
  auto vy = st.column<float>("vel_y");
  auto hp = st.column<float>("health");
  auto stt= st.column<uint32_t>("state");
  for (uint32_t r=0; r<st.size(); ++r) {
    px[r]=0.f; py[r]=0.f; vx[r]=1.f; vy[r]=0.5f; hp[r]=100.f; stt[r]=(r&1u)?1u:0u;
  }

  if (!det) {
    // in-place fast path (use k.idx for logical→physical)
    pipe.addMap("integrate",[&](KernelCtx& k){
      auto px = st.column<float>("pos_x");
      auto py = st.column<float>("pos_y");
      auto vx = st.column<float>("vel_x");
      auto vy = st.column<float>("vel_y");
      auto stt= st.column<uint32_t>("state");
      const float dt = 0.016f;
      for (uint32_t r=k.tileBegin; r<k.tileEnd; ++r) {
        uint32_t rr = k.idx ? k.idx[r] : r;
        if (stt[rr]==1u){ px[rr] += vx[rr]*dt; py[rr] += vy[rr]*dt; }
      }
    });
    pipe.addMap("damage",[&](KernelCtx& k){
      auto hp = st.column<float>("health");
      auto stt= st.column<uint32_t>("state");
      for (uint32_t r=k.tileBegin; r<k.tileEnd; ++r) {
        uint32_t rr = k.idx ? k.idx[r] : r;
        if (stt[rr]==1u){ float h = hp[rr]-0.1f; hp[rr] = h>0.f? h:0.f; }
      }
    });
  } else {
    // deterministic Sow path (rows are logical; resolveAndCommit maps to physical)
    pipe.addMap("integrate",[&](KernelCtx& k){
      auto vx = st.column<float>("vel_x");
      auto vy = st.column<float>("vel_y");
      auto stt= st.column<uint32_t>("state");
      const float dt = 0.016f;
      auto& pxw = k.sow->fieldF32("pos_x");
      auto& pyw = k.sow->fieldF32("pos_y");
      for (uint32_t r=k.tileBegin; r<k.tileEnd; ++r) {
        // logical r is fine here; commit will map
        if (stt[r]==1u){ pxw.add[r] += vx[r]*dt; pyw.add[r] += vy[r]*dt; }
      }
    });
    pipe.addMap("damage",[&](KernelCtx& k){
      auto hpR = st.column<float>("health");
      auto stt = st.column<uint32_t>("state");
      auto& hpw = k.sow->fieldF32("health");
      for (uint32_t r=k.tileBegin; r<k.tileEnd; ++r) {
        if (stt[r]==1u){ float h = hpR[r]-0.1f; if (h < 0.f) h = 0.f; hpw.set[r] = h; }
      }
    });
  }
}

// --- Finance kernels (SoA)
static void build_finance(Store& st, Pipeline& pipe, bool det) {
  auto px = st.column<float>("price");
  auto q  = st.column<float>("qty");
  auto sd = st.column<uint32_t>("side");
  for (uint32_t r=0; r<st.size(); ++r) {
    px[r] = 100.0f + (r % 100) * 0.01f;
    q[r]  = 1.0f;
    sd[r] = (r & 1u);
  }

  if (!det) {
    pipe.addMap("price_band_update",[&](KernelCtx& k){
      auto q  = st.column<float>("qty");
      auto sd = st.column<uint32_t>("side");
      for (uint32_t r=k.tileBegin; r<k.tileEnd; ++r) {
        uint32_t rr = k.idx ? k.idx[r] : r;
        q[rr] += (sd[rr]==0u) ? 1.0f : -1.0f;
        if (q[rr] < 0.f) q[rr] = 0.f;
      }
    });
  } else {
    pipe.addMap("price_band_update",[&](KernelCtx& k){
      auto sd = st.column<uint32_t>("side");
      auto& qw = k.sow->fieldF32("qty");
      for (uint32_t r=k.tileBegin; r<k.tileEnd; ++r) {
        float delta = (sd[r]==0u) ? 1.0f : -1.0f; // logical r; commit will map
        qw.add[r] += delta;
      }
    });
  }
}

// --- AoS control (unchanged)
struct Particle { float x,y; float vx,vy; float health; uint32_t state; };

static void run_aos_gaming(uint32_t n, uint32_t ticks, std::vector<double>& samples) {
  std::vector<Particle> P(n);
  for (uint32_t i=0; i<n; ++i) P[i] = {0.f,0.f,1.f,0.5f,100.f,(i&1u)?1u:0u};
  using H = std::chrono::high_resolution_clock;
  for (uint32_t t=0; t<ticks; ++t) {
    auto t0 = H::now();
    const float dt=0.016f;
    for (uint32_t i=0; i<n; ++i) if (P[i].state==1u){ P[i].x += P[i].vx*dt; P[i].y += P[i].vy*dt; }
    for (uint32_t i=0; i<n; ++i) if (P[i].state==1u){ float h=P[i].health-0.1f; P[i].health = h>0.f?h:0.f; }
    auto t1 = H::now();
    samples.push_back(std::chrono::duration<double,std::micro>(t1 - t0).count());
  }
}

static void run_aos_finance(uint32_t n, uint32_t ticks, std::vector<double>& samples) {
  struct Order{ float price, qty; uint32_t side, ts; };
  std::vector<Order> B(n);
  for (uint32_t i=0; i<n; ++i) B[i] = {100.f + (i%100)*0.01f, 1.0f, (i&1u), i};
  using H = std::chrono::high_resolution_clock;
  for (uint32_t t=0; t<ticks; ++t) {
    auto t0 = H::now();
    for (uint32_t i=0; i<n; ++i) { float d = (B[i].side==0u)? 1.f : -1.f; B[i].qty += d; if (B[i].qty<0.f) B[i].qty=0.f; }
    auto t1 = H::now();
    samples.push_back(std::chrono::duration<double,std::micro>(t1 - t0).count());
  }
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);

  #ifdef OOMX_WITH_CUDA
  if (std::getenv("OOMX_GPU")){
    using namespace oomx;
    if (a.sim=="gaming"){
      // build host SoA using Store, then upload
      Schema s; s.add({"pos_x",FieldType::F32,0}).add({"pos_y",FieldType::F32,0})
                .add({"vel_x",FieldType::F32,0}).add({"vel_y",FieldType::F32,0})
                .add({"health",FieldType::F32,0}).add({"state",FieldType::U32,0});
      Store st{s, TileConfig{2048,32,true}}; for(uint32_t i=0;i<a.n;++i) st.create();
      auto px=st.column<float>("pos_x"), py=st.column<float>("pos_y"),
          vx=st.column<float>("vel_x"), vy=st.column<float>("vel_y"),
          hp=st.column<float>("health"); auto stt=st.column<uint32_t>("state");
      for(uint32_t r=0;r<st.size();++r){ px[r]=0; py[r]=0; vx[r]=1; vy[r]=0.5f; hp[r]=100; stt[r]=(r&1u)?1u:0u; }
      GpuBuffers gb; gpuAllocGaming(gb, a.n); gpuUploadGaming(gb, px,py,vx,vy,hp,stt);
      auto tim = gpuRunGaming(gb, a.ticks);
      std::printf("GPU gaming: N=%u iters=%u kernel=%.3f ms (%.3f us/iter)",
                  a.n, a.ticks, tim.kernel_ms, 1000.0*tim.kernel_ms/a.ticks);
                  gpuDownloadGaming(gb, px,py,hp); gpuFree(gb); return 0;
    } else {
      Schema s; s.add({"price",FieldType::F32,0}).add({"qty",FieldType::F32,0})
                .add({"side",FieldType::U32,0}).add({"ts",FieldType::U32,0});
      Store st{s, TileConfig{2048,32,true}}; for(uint32_t i=0;i<a.n;++i) st.create();
      auto pr=st.column<float>("price"), qt=st.column<float>("qty"); auto sd=st.column<uint32_t>("side");
      for(uint32_t r=0;r<st.size();++r){ pr[r]=100.f+(r%100)*0.01f; qt[r]=1.f; sd[r]=(r&1u); }
      GpuBuffers gb; gpuAllocFinance(gb, a.n); gpuUploadFinance(gb, pr,qt,sd);
      auto tim = gpuRunFinance(gb, a.ticks);
      std::printf("GPU finance: N=%u iters=%u kernel=%.3f ms (%.3f us/iter)",
                  a.n, a.ticks, tim.kernel_ms, 1000.0*tim.kernel_ms/a.ticks);
                  gpuDownloadFinance(gb, qt); gpuFree(gb); return 0;
    }
  }
  #endif

  // AoS control group
  if (a.mode == "aos") {
    std::vector<double> samples; samples.reserve(a.ticks);
    if (a.sim=="gaming") run_aos_gaming(a.n, a.ticks, samples);
    else                 run_aos_finance(a.n, a.ticks, samples);
    stats_print("aos", samples);
    write_csv(a.csv, samples);
    return 0;
  }

  // SoA / AoSoA
  Schema s;
  if (a.sim=="gaming") {
    s.add({"pos_x",FieldType::F32,0}).add({"pos_y",FieldType::F32,0})
     .add({"vel_x",FieldType::F32,0}).add({"vel_y",FieldType::F32,0})
     .add({"health",FieldType::F32,0}).add({"state",FieldType::U32,0});
  } else {
    s.add({"price",FieldType::F32,0}).add({"qty",FieldType::F32,0})
     .add({"side",FieldType::U32,0}).add({"ts",FieldType::U32,0});
  }

  Store st{s, TileConfig{a.tile, 32, true}};
  for (uint32_t i=0; i<a.n; ++i) st.create();

  Runtime rt{&st};
  rt.setTileRows(a.tile);
  rt.setDeterministic(a.det);
  rt.setDynamic(a.dynamic);
  rt.enableIndexMode(true); // <<— zero-copy dynamic regrouping
  rt.setPolicy(Policy{.min_branch=0.85, .min_coalescing=0.90, .windows=3});

  Pipeline pipe;
  if (a.sim=="gaming") build_gaming(st, pipe, a.det);
  else                 build_finance(st, pipe, a.det);

  // warmup (unmeasured) — lets dynamic index build happen off-clock
  for (uint32_t t=0; t<a.warmup; ++t) {
    rt.run(pipe);
    rt.resolveAndCommit();
  }

  // measured loop
  std::vector<double> samples; samples.reserve(a.ticks);
  using H = std::chrono::high_resolution_clock;
  for (uint32_t t=0; t<a.ticks; ++t) {
    auto t0 = H::now();
    rt.run(pipe);
    rt.resolveAndCommit();
    auto t1 = H::now();
    samples.push_back(std::chrono::duration<double,std::micro>(t1 - t0).count());
  }

  write_csv(a.csv, samples);
  auto m = rt.metrics();
  std::printf("sim=%s mode=%s dynamic=%d n=%u ticks=%u warmup=%u tile=%u mean=%.2fus p95=%.2fus p99=%.2fus\n",
              a.sim.c_str(), a.mode.c_str(), a.dynamic?1:0, a.n, a.ticks, a.warmup, a.tile,
              m.mean_us, m.p95_us, m.p99_us);

  return 0;
}
