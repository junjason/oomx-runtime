#include "oomx/Gpu.h"
#include <cuda_runtime.h>
#include <cstdio>


namespace oomx {


// --- error macro
#define CUCHK(stmt) do{ cudaError_t err=(stmt); if(err!=cudaSuccess){
std::fprintf(stderr,"CUDA error %s:%d: %s
", __FILE__, __LINE__, cudaGetErrorString(err));
std::abort(); } } while(0)


// --- kernels (coalesced SoA)
__global__ void k_integrate(float* __restrict__ px, float* __restrict__ py,
const float* __restrict__ vx, const float* __restrict__ vy,
const uint32_t* __restrict__ state, float dt, uint32_t N){
uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i>=N) return;
if (state[i]==1u){ px[i] += vx[i]*dt; py[i] += vy[i]*dt; }
}


__global__ void k_damage(float* __restrict__ hp, const uint32_t* __restrict__ state, uint32_t N){
uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i>=N) return;
if (state[i]==1u){ float h=hp[i]-0.1f; hp[i] = h>0.f ? h : 0.f; }
}


__global__ void k_price_band(float* __restrict__ qty, const uint32_t* __restrict__ side, uint32_t N){
uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i>=N) return;
float d = (side[i]==0u)? 1.f : -1.f;
float q = qty[i] + d; qty[i] = (q>0.f)? q : 0.f;
}


static dim3 gridFor(uint32_t N, uint32_t block=256){ return dim3((N + block - 1)/block); }


// --- alloc/upload/download
void gpuAllocGaming(GpuBuffers& b, uint32_t N){ b.N=N;
CUCHK(cudaMalloc(&b.d_pos_x, N*sizeof(float)));
CUCHK(cudaMalloc(&b.d_pos_y, N*sizeof(float)));
CUCHK(cudaMalloc(&b.d_vel_x, N*sizeof(float)));
CUCHK(cudaMalloc(&b.d_vel_y, N*sizeof(float)));
CUCHK(cudaMalloc(&b.d_health, N*sizeof(float)));
CUCHK(cudaMalloc(&b.d_state, N*sizeof(uint32_t)));
}


void gpuAllocFinance(GpuBuffers& b, uint32_t N){ b.N=N;
CUCHK(cudaMalloc(&b.d_price, N*sizeof(float)));
CUCHK(cudaMalloc(&b.d_qty, N*sizeof(float)));
CUCHK(cudaMalloc(&b.d_side, N*sizeof(uint32_t)));
}

void gpuUploadGaming(GpuBuffers& b, const float* px, const float* py,
const float* vx, const float* vy,
const float* hp, const uint32_t* st){
CUCHK(cudaMemcpy(b.d_pos_x, px, b.N*sizeof(float), cudaMemcpyHostToDevice));
CUCHK(cudaMemcpy(b.d_pos_y, py, b.N*sizeof(float), cudaMemcpyHostToDevice));
CUCHK(cudaMemcpy(b.d_vel_x, vx, b.N*sizeof(float), cudaMemcpyHostToDevice));
CUCHK(cudaMemcpy(b.d_vel_y, vy, b.N*sizeof(float), cudaMemcpyHostToDevice));
CUCHK(cudaMemcpy(b.d_health, hp, b.N*sizeof(float), cudaMemcpyHostToDevice));
CUCHK(cudaMemcpy(b.d_state, st, b.N*sizeof(uint32_t), cudaMemcpyHostToDevice));
}


void gpuUploadFinance(GpuBuffers& b, const float* price, const float* qty, const uint32_t* side){
CUCHK(cudaMemcpy(b.d_price, price, b.N*sizeof(float), cudaMemcpyHostToDevice));
CUCHK(cudaMemcpy(b.d_qty, qty, b.N*sizeof(float), cudaMemcpyHostToDevice));
CUCHK(cudaMemcpy(b.d_side, side, b.N*sizeof(uint32_t), cudaMemcpyHostToDevice));
}


void gpuDownloadGaming(const GpuBuffers& b, float* px, float* py, float* hp){
CUCHK(cudaMemcpy(px, b.d_pos_x, b.N*sizeof(float), cudaMemcpyDeviceToHost));
CUCHK(cudaMemcpy(py, b.d_pos_y, b.N*sizeof(float), cudaMemcpyDeviceToHost));
CUCHK(cudaMemcpy(hp, b.d_health, b.N*sizeof(float), cudaMemcpyDeviceToHost));
}


void gpuDownloadFinance(const GpuBuffers& b, float* qty){
CUCHK(cudaMemcpy(qty, b.d_qty, b.N*sizeof(float), cudaMemcpyDeviceToHost));
}


void gpuFree(GpuBuffers& b){
auto fre = [&](void* p){ if(p) cudaFree(p); };
fre(b.d_pos_x); fre(b.d_pos_y); fre(b.d_vel_x); fre(b.d_vel_y); fre(b.d_health); fre(b.d_state);
fre(b.d_price); fre(b.d_qty); fre(b.d_side);
b = {};
}


// --- runs
GpuTiming gpuRunGaming(const GpuBuffers& b, uint32_t iters){
const uint32_t block=256; auto grid=gridFor(b.N,block);
cudaEvent_t a,beg,end; CUCHK(cudaEventCreate(&a)); CUCHK(cudaEventCreate(&beg)); CUCHK(cudaEventCreate(&end));
float ms=0.f; const float dt=0.016f;
CUCHK(cudaEventRecord(beg));
for(uint32_t t=0;t<iters;++t){
k_integrate<<<grid,block>>>(b.d_pos_x,b.d_pos_y,b.d_vel_x,b.d_vel_y,b.d_state,dt,b.N);
k_damage<<<grid,block>>>(b.d_health,b.d_state,b.N);
}
CUCHK(cudaEventRecord(end)); CUCHK(cudaEventSynchronize(end));
CUCHK(cudaEventElapsedTime(&ms, beg, end));
CUCHK(cudaEventDestroy(a)); CUCHK(cudaEventDestroy(beg)); CUCHK(cudaEventDestroy(end));
return { .kernel_ms = ms };
}

GpuTiming gpuRunFinance(const GpuBuffers& b, uint32_t iters){
const uint32_t block=256; auto grid=gridFor(b.N,block);
cudaEvent_t beg,end; CUCHK(cudaEventCreate(&beg)); CUCHK(cudaEventCreate(&end));
float ms=0.f;
CUCHK(cudaEventRecord(beg));
for(uint32_t t=0;t<iters;++t){
k_price_band<<<grid,block>>>(b.d_qty,b.d_side,b.N);
}
CUCHK(cudaEventRecord(end)); CUCHK(cudaEventSynchronize(end));
CUCHK(cudaEventElapsedTime(&ms, beg, end));
CUCHK(cudaEventDestroy(beg)); CUCHK(cudaEventDestroy(end));
return { .kernel_ms = ms };
}


} // namespace oomx