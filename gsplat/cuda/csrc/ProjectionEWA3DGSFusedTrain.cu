#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_fwd_train_kernel(
	const uint32_t C,
	const uint32_t N,
	const scalar_t *__restrict__ means,
	const scalar_t *__restrict__ covars,
	const scalar_t *__restrict__ quats,
	const scalar_t *__restrict__ scales,
	const scalar_t *__restrict__ viewmats,
	const scalar_t *__restrict__ Ks,
	const int32_t image_width,
	const int32_t image_height,
	const float eps2d,
	const float near_plane,
	const float far_plane,
	const float radius_clip,
	const CameraModelType camera_model,
	int32_t *__restrict__ radii,
	scalar_t *__restrict__ means2d,
	scalar_t *__restrict__ depths,
	scalar_t *__restrict__ conics,
	scalar_t *__restrict__ compensations
) {
	uint32_t idx = cg::this_grid().thread_rank();
	if (idx >= C * N) {
		return;
	}
	const uint32_t cid = idx / N;
	const uint32_t gid = idx % N;

	means += gid * 3;
	viewmats += cid * 16;
	Ks += cid * 9;

	mat3 R = mat3(
		viewmats[0], viewmats[4], viewmats[8],
		viewmats[1], viewmats[5], viewmats[9],
		viewmats[2], viewmats[6], viewmats[10]
	);
	vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

	vec3 mean_c;
	posW2C(R, t, glm::make_vec3(means), mean_c);
	if (mean_c.z < near_plane || mean_c.z > far_plane) {
		radii[idx] = 0;
		return;
	}

	mat3 covar;
	if (covars != nullptr) {
		covars += gid * 6;
		covar = mat3(
			covars[0], covars[1], covars[2],
			covars[1], covars[3], covars[4],
			covars[2], covars[4], covars[5]
		);
	} else {
		quats += gid * 4;
		scales += gid * 3;
		quat_scale_to_covar_preci(
			glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
		);
	}
	mat3 covar_c;
	covarW2C(R, covar, covar_c);

	mat2 covar2d;
	vec2 mean2d;
	switch (camera_model) {
	case CameraModelType::PINHOLE:
		persp_proj(
			mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5],
			image_width, image_height, covar2d, mean2d
		);
		break;
	case CameraModelType::ORTHO:
		ortho_proj(
			mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5],
			image_width, image_height, covar2d, mean2d
		);
		break;
	case CameraModelType::FISHEYE:
		fisheye_proj(
			mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5],
			image_width, image_height, covar2d, mean2d
		);
		break;
	}

	float compensation;
	float det = add_blur(eps2d, covar2d, compensation);
	if (det <= 0.f) {
		radii[idx] = 0;
		return;
	}

	mat2 covar2d_inv = glm::inverse(covar2d);

	float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
	float v1 = b + sqrt(max(0.01f, b * b - det));
	float radius = ceil(3.f * sqrt(v1));

	if (radius <= radius_clip) {
		radii[idx] = 0;
		return;
	}

	if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
		mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
		radii[idx] = 0;
		return;
	}

	radii[idx] = (int32_t)radius;
	means2d[idx * 2] = mean2d.x;
	means2d[idx * 2 + 1] = mean2d.y;
	depths[idx] = mean_c.z;
	// Training/original parameterization.
	conics[idx * 3] = covar2d_inv[0][0];
	conics[idx * 3 + 1] = covar2d_inv[0][1];
	conics[idx * 3 + 2] = covar2d_inv[1][1];
	if (compensations != nullptr) {
		compensations[idx] = compensation;
	}
}

void launch_projection_ewa_3dgs_fused_fwd_train_kernel(
	const at::Tensor means,
	const at::optional<at::Tensor> covars,
	const at::optional<at::Tensor> quats,
	const at::optional<at::Tensor> scales,
	const at::Tensor viewmats,
	const at::Tensor Ks,
	const uint32_t image_width,
	const uint32_t image_height,
	const float eps2d,
	const float near_plane,
	const float far_plane,
	const float radius_clip,
	const CameraModelType camera_model,
	at::Tensor radii,
	at::Tensor means2d,
	at::Tensor depths,
	at::Tensor conics,
	at::optional<at::Tensor> compensations
) {
	uint32_t N = means.size(0);
	uint32_t C = viewmats.size(0);

	int64_t n_elements = C * N;
	dim3 threads(256);
	dim3 grid((n_elements + threads.x - 1) / threads.x);
	int64_t shmem_size = 0;

	if (n_elements == 0) {
		return;
	}

	AT_DISPATCH_FLOATING_TYPES(
		means.scalar_type(),
		"projection_ewa_3dgs_fused_fwd_train_kernel",
		[&]() {
			projection_ewa_3dgs_fused_fwd_train_kernel<scalar_t>
				<<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
					C,
					N,
					means.data_ptr<scalar_t>(),
					covars.has_value() ? covars.value().data_ptr<scalar_t>() : nullptr,
					quats.has_value() ? quats.value().data_ptr<scalar_t>() : nullptr,
					scales.has_value() ? scales.value().data_ptr<scalar_t>() : nullptr,
					viewmats.data_ptr<scalar_t>(),
					Ks.data_ptr<scalar_t>(),
					image_width,
					image_height,
					eps2d,
					near_plane,
					far_plane,
					radius_clip,
					camera_model,
					radii.data_ptr<int32_t>(),
					means2d.data_ptr<scalar_t>(),
					depths.data_ptr<scalar_t>(),
					conics.data_ptr<scalar_t>(),
					compensations.has_value()
						? compensations.value().data_ptr<scalar_t>()
						: nullptr
				);
		}
	);
}

} // namespace gsplat

// Reuse backward implementation while exposing train-symbol names.
#define projection_ewa_3dgs_fused_fwd_kernel projection_ewa_3dgs_fused_fwd_train_shadow_kernel
#define launch_projection_ewa_3dgs_fused_fwd_kernel launch_projection_ewa_3dgs_fused_fwd_train_shadow_kernel
#define projection_ewa_3dgs_fused_bwd_kernel projection_ewa_3dgs_fused_bwd_train_kernel
#define launch_projection_ewa_3dgs_fused_bwd_kernel launch_projection_ewa_3dgs_fused_bwd_train_kernel
#include "ProjectionEWA3DGSFused.cu"
#undef projection_ewa_3dgs_fused_fwd_kernel
#undef launch_projection_ewa_3dgs_fused_fwd_kernel
#undef projection_ewa_3dgs_fused_bwd_kernel
#undef launch_projection_ewa_3dgs_fused_bwd_kernel
