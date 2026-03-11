#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_train_kernel(
	const uint32_t C,
	const uint32_t N,
	const uint32_t n_isects,
	const bool packed,
	const vec2 *__restrict__ means2d,
	const vec3 *__restrict__ conics,
	const scalar_t *__restrict__ colors,
	const scalar_t *__restrict__ opacities,
	const scalar_t *__restrict__ backgrounds,
	const bool *__restrict__ masks,
	const uint32_t image_width,
	const uint32_t image_height,
	const uint32_t tile_size,
	const uint32_t tile_width,
	const uint32_t tile_height,
	const int32_t *__restrict__ tile_offsets,
	const int32_t *__restrict__ flatten_ids,
	scalar_t *__restrict__ render_colors,
	scalar_t *__restrict__ render_alphas,
	int32_t *__restrict__ last_ids
) {
	auto block = cg::this_thread_block();
	int32_t camera_id = block.group_index().x;
	int32_t tile_id =
		block.group_index().y * tile_width + block.group_index().z;
	uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
	uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

	tile_offsets += camera_id * tile_height * tile_width;
	render_colors += camera_id * image_height * image_width * CDIM;
	render_alphas += camera_id * image_height * image_width;
	last_ids += camera_id * image_height * image_width;
	if (backgrounds != nullptr) {
		backgrounds += camera_id * CDIM;
	}
	if (masks != nullptr) {
		masks += camera_id * tile_height * tile_width;
	}

	float px = (float)j + 0.5f;
	float py = (float)i + 0.5f;
	int32_t pix_id = i * image_width + j;

	bool inside = (i < image_height && j < image_width);
	bool done = !inside;

	if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
		for (uint32_t k = 0; k < CDIM; ++k) {
			render_colors[pix_id * CDIM + k] =
				backgrounds == nullptr ? 0.0f : backgrounds[k];
		}
		return;
	}

	int32_t range_start = tile_offsets[tile_id];
	int32_t range_end =
		(camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
			? n_isects
			: tile_offsets[tile_id + 1];
	const uint32_t block_size = block.size();
	uint32_t num_batches =
		(range_end - range_start + block_size - 1) / block_size;

	extern __shared__ int s[];
	int32_t *id_batch = (int32_t *)s;
	vec3 *xy_opacity_batch =
		reinterpret_cast<vec3 *>(&id_batch[block_size]);
	vec3 *conic_batch =
		reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);

	float T = 1.0f;
	uint32_t cur_idx = 0;
	uint32_t tr = block.thread_rank();

	float pix_out[CDIM] = {0.f};
	for (uint32_t b = 0; b < num_batches; ++b) {
		if (__syncthreads_count(done) >= block_size) {
			break;
		}

		uint32_t batch_start = range_start + block_size * b;
		uint32_t idx = batch_start + tr;
		if (idx < range_end) {
			int32_t g = flatten_ids[idx];
			id_batch[tr] = g;
			const vec2 xy = means2d[g];
			const float opac = opacities[g];
			xy_opacity_batch[tr] = {xy.x, xy.y, opac};
			conic_batch[tr] = conics[g];
		}

		block.sync();

		uint32_t batch_size = min(block_size, range_end - batch_start);
		for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
			const vec3 conic = conic_batch[t];
			const vec3 xy_opac = xy_opacity_batch[t];
			const float opac = xy_opac.z;
			const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
			const float sigma = 0.5f * (conic.x * delta.x * delta.x +
										conic.z * delta.y * delta.y) +
								conic.y * delta.x * delta.y;
			float alpha = min(0.999f, opac * expf(-sigma));
			if (sigma < 0.f || alpha < 1.f / 255.f) {
				continue;
			}

			const float next_T = T * (1.0f - alpha);
			if (next_T <= 1e-4f) {
				done = true;
				break;
			}

			int32_t g = id_batch[t];
			const float vis = alpha * T;
			const float *c_ptr = colors + g * CDIM;
#pragma unroll
			for (uint32_t k = 0; k < CDIM; ++k) {
				pix_out[k] += c_ptr[k] * vis;
			}
			cur_idx = batch_start + t;
			T = next_T;
		}
	}

	if (inside) {
		render_alphas[pix_id] = 1.0f - T;
#pragma unroll
		for (uint32_t k = 0; k < CDIM; ++k) {
			render_colors[pix_id * CDIM + k] =
				backgrounds == nullptr ? pix_out[k]
									   : (pix_out[k] + T * backgrounds[k]);
		}
		last_ids[pix_id] = static_cast<int32_t>(cur_idx);
	}
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_train_kernel(
	const at::Tensor means2d,
	const at::Tensor conics,
	const at::Tensor colors,
	const at::Tensor opacities,
	const at::optional<at::Tensor> backgrounds,
	const at::optional<at::Tensor> masks,
	const uint32_t image_width,
	const uint32_t image_height,
	const uint32_t tile_size,
	const at::Tensor tile_offsets,
	const at::Tensor flatten_ids,
	at::Tensor renders,
	at::Tensor alphas,
	at::Tensor last_ids
) {
	bool packed = means2d.dim() == 2;

	uint32_t C = tile_offsets.size(0);
	uint32_t N = packed ? 0 : means2d.size(1);
	uint32_t tile_height = tile_offsets.size(1);
	uint32_t tile_width = tile_offsets.size(2);
	uint32_t n_isects = flatten_ids.size(0);

	dim3 threads = {tile_size, tile_size, 1};
	dim3 grid = {C, tile_height, tile_width};

	int64_t shmem_size =
		tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

	if (cudaFuncSetAttribute(
			rasterize_to_pixels_3dgs_fwd_train_kernel<CDIM, float>,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			shmem_size
		) != cudaSuccess) {
		AT_ERROR(
			"Failed to set maximum shared memory size (requested ",
			shmem_size,
			" bytes), try lowering tile_size."
		);
	}

	rasterize_to_pixels_3dgs_fwd_train_kernel<CDIM, float>
		<<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
			C,
			N,
			n_isects,
			packed,
			reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
			reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
			colors.data_ptr<float>(),
			opacities.data_ptr<float>(),
			backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
									: nullptr,
			masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
			image_width,
			image_height,
			tile_size,
			tile_width,
			tile_height,
			tile_offsets.data_ptr<int32_t>(),
			flatten_ids.data_ptr<int32_t>(),
			renders.data_ptr<float>(),
			alphas.data_ptr<float>(),
			last_ids.data_ptr<int32_t>()
		);
}

#define __INS_TRAIN__(CDIM)                                                    \
	template void launch_rasterize_to_pixels_3dgs_fwd_train_kernel<CDIM>(      \
		const at::Tensor means2d,                                              \
		const at::Tensor conics,                                               \
		const at::Tensor colors,                                               \
		const at::Tensor opacities,                                            \
		const at::optional<at::Tensor> backgrounds,                            \
		const at::optional<at::Tensor> masks,                                  \
		uint32_t image_width,                                                  \
		uint32_t image_height,                                                 \
		uint32_t tile_size,                                                    \
		const at::Tensor tile_offsets,                                         \
		const at::Tensor flatten_ids,                                          \
		at::Tensor renders,                                                    \
		at::Tensor alphas,                                                     \
		at::Tensor last_ids                                                    \
	);

__INS_TRAIN__(1)
__INS_TRAIN__(2)
__INS_TRAIN__(3)
__INS_TRAIN__(4)
__INS_TRAIN__(5)
__INS_TRAIN__(8)
__INS_TRAIN__(9)
__INS_TRAIN__(16)
__INS_TRAIN__(17)
__INS_TRAIN__(32)
__INS_TRAIN__(33)
__INS_TRAIN__(64)
__INS_TRAIN__(65)
__INS_TRAIN__(128)
__INS_TRAIN__(129)
__INS_TRAIN__(256)
__INS_TRAIN__(257)
__INS_TRAIN__(512)
__INS_TRAIN__(513)
#undef __INS_TRAIN__

} // namespace gsplat
