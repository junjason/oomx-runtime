#include "oomx/Schema.h"
#include "oomx/Store.h"
#include <cstdio>
#include <cuda_runtime.h>

using namespace oomx;

__global__ void integrateKernel(float* px,float* py,const float* vx,const float* vy,
                                const uint32_t* state,const uint32_t* idx,
                                uint32_t begin,uint32_t end,float dt) {
  uint32_t r = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= end) return;
  uint32_t rr = idx ? idx[r] : r;
  if (state[rr]==1u) { px[rr] += vx[rr]*dt; py[rr] += vy[rr]*dt; }
}

__global__ void damageKernel(float* hp,const uint32_t* state,const uint32_t* idx,
                             uint32_t begin,uint32_t end) {
  uint32_t r = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= end) return;
  uint32_t rr = idx ? idx[r] : r;
  if (state[rr]==1u) { float h = hp[rr]-0.1f; hp[rr] = h>0.f? h:0.f; }
}

int main(int argc,char**){
  // Build gaming schema + data on host
  Schema s;
  s.add({"pos_x",FieldType::F32,0}).add({"pos_y",FieldType::F32,0})
   .add({"vel_x",FieldType::F32,0}).add({"vel_y",FieldType::F32,0})
   .add({"health",FieldType::F32,0}).add({"state",FieldType::U32,0});

  const uint32_t N = 200000;
  Store st{s, {/*tileRows*/8192, /*lane*/32, /*aosoa*/true}};
  for (uint32_t i=0;i<N;++i) st.create();

  auto px = st.column<float>("pos_x");
  auto py = st.column<float>("pos_y");
  auto vx = st.column<float>("vel_x");
  auto vy = st.column<float>("vel_y");
  auto hp = st.column<float>("health");
  auto stt= st.column<uint32_t>("state");
  for (uint32_t r=0;r<N;++r){ px[r]=0; py[r]=0; vx[r]=1; vy[r]=0.5f; hp[r]=100; stt[r]=(r&1u)?1u:0u; }

  // Device buffers (simple pinned-copy MVP)
  float *d_px,*d_py,*d_vx,*d_vy,*d_hp; uint32_t *d_state; uint32_t *d_idx=nullptr;
  cudaMalloc(&d_px,N*sizeof(float)); cudaMalloc(&d_py,N*sizeof(float));
  cudaMalloc(&d_vx,N*sizeof(float)); cudaMalloc(&d_vy,N*sizeof(float));
  cudaMalloc(&d_hp,N*sizeof(float)); cudaMalloc(&d_state,N*sizeof(uint32_t));
  cudaMemcpy(d_px,px,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_py,py,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_vx,vx,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy,vy,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_hp,hp,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_state,stt,N*sizeof(uint32_t),cudaMemcpyHostToDevice);

  // Optional: build ACTIVE-first idx on host and upload (zero-copy “dynamic”)
  // Omit for first run (pass nullptr).
  // If you want it: allocate host idx, fill with active-first order, cudaMalloc d_idx, cudaMemcpy.

  dim3 block(256);
  dim3 grid((N + block.x - 1)/block.x);
  const float dt = 0.016f;

  // Simple timing over T ticks
  const int T = 100;
  cudaDeviceSynchronize();
  for(int t=0;t<T;++t){
    integrateKernel<<<grid,block>>>(d_px,d_py,d_vx,d_vy,d_state,d_idx,0,N,dt);
    damageKernel<<<grid,block>>>(d_hp,d_state,d_idx,0,N);
  }
  cudaDeviceSynchronize();

  // Copy one value back to ensure side-effects happened
  cudaMemcpy(px,d_px,1*sizeof(float),cudaMemcpyDeviceToHost);
  printf("px[0]=%f\n", px[0]);

  // Cleanup
  cudaFree(d_px); cudaFree(d_py); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_hp); cudaFree(d_state);
  if (d_idx) cudaFree(d_idx);
  return 0;
}
