#pragma once
#include <cstdint>
#include <vector>
#include <string>


namespace oomx {
struct GpuBuffers {
	// raw device pointers (SoA)
	float *d_pos_x{}, *d_pos_y{}, *d_vel_x{}, *d_vel_y{}, *d_health{};
	uint32_t *d_state{};
	// finance
	float *d_price{}, *d_qty{}; uint32_t *d_side{};
	uint32_t N{};
	};


	struct GpuTiming { float kernel_ms=0.f; };


	// Upload/Download helpers
	void gpuAllocGaming(GpuBuffers& b, uint32_t N);
	void gpuUploadGaming(GpuBuffers& b,
	const float* px, const float* py,
	const float* vx, const float* vy,
	const float* hp, const uint32_t* st);
	void gpuDownloadGaming(const GpuBuffers& b,
	float* px, float* py, float* hp);
	void gpuFree(GpuBuffers& b);


	// Kernels (single stream)
	GpuTiming gpuRunGaming(const GpuBuffers& b, uint32_t iters);
	GpuTiming gpuRunFinance(const GpuBuffers& b, uint32_t iters);


	// Finance helpers
	void gpuAllocFinance(GpuBuffers& b, uint32_t N);
	void gpuUploadFinance(GpuBuffers& b,
	const float* price, const float* qty, const uint32_t* side);
	void gpuDownloadFinance(const GpuBuffers& b, float* qty);
	}