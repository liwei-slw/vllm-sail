// SPDX-License-Identifier: Apache-2.0
#pragma once

// sailify rewrites upstream's __CUDA_ARCH__, __CUDA_ARCH_LIST__ and CUDA_VERSION
// gates into the names below but leaves them undefined. This header supplies the
// values a SAIL torch's .ppu_compat/compatible_wrapper.h provided, so the ported
// corpus builds without that wheel header. Values match sailify's defaults.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// The wrapper this replaces was force-included into every ported source and
// covered their missing standard includes; cuda_utils.h, for example, uses
// std::enable_if_t and int64_t with only <stdio.h>.
#if defined(__cplusplus)
#include <type_traits>
#endif

// ...and its SDK umbrella: cuda_utils_kernels.cu expands CUDA_CHECK's
// hggcError_t with no SDK header of its own. Key on the HG compiler marker
// rather than __has_include, which the PPU preprocessor does not reliably
// evaluate; host-only compiles still probe so they need no PPU SDK.
#if defined(__HGGCCC__)
#define PPU_COMPAT_HAS_SDK
#elif defined(__has_include)
#if __has_include(<hggc.h>)
#define PPU_COMPAT_HAS_SDK
#endif
#endif

#ifdef PPU_COMPAT_HAS_SDK
#include <hggc.h>
#if defined(__cplusplus)
#include <acblas_v2.h>
#endif
#endif

// The CUDA-compatible nvcc driver defines __CUDACC__ alongside __HGGCCC__.
// A SAIL torch switches to the SDK's scalar types under __HGGCCC__; a stock
// torch still takes its __CUDACC__ branch and includes NVIDIA's cuda_fp16.h
// and cuda_bf16.h. Those define a second __half/__nv_bfloat16 universe next to
// hggc_fp16.h/hggc_bf16.h, so ported sources that include the native headers
// collide with thousands of redefinition and overload errors. Take the native
// headers first, mask NVIDIA's via their include guards, and give the NVIDIA
// bf16 spellings the native types so stock torch headers compile unchanged.
#if defined(__HGGCCC__) && defined(__cplusplus)
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <hggc_runtime_api.h>
#include <hggc_runtime_api.hpp>
#define __CUDA_FP16_H__
#define __CUDA_BF16_H__
typedef __ppu_bfloat16 __nv_bfloat16;
typedef __ppu_bfloat162 __nv_bfloat162;
typedef __ppu_bfloat16_raw __nv_bfloat16_raw;
typedef __ppu_bfloat162_raw __nv_bfloat162_raw;
#endif

#ifndef COMPATIBLE_VERSION
#define COMPATIBLE_VERSION 13000
#endif

#ifndef COMPATIBLE_RT_VERSION
#define COMPATIBLE_RT_VERSION COMPATIBLE_VERSION
#endif

#ifndef __COMPATIBLECC_VER_MAJOR__
#define __COMPATIBLECC_VER_MAJOR__ 13
#endif
#ifndef __COMPATIBLECC_VER_MINOR__
#define __COMPATIBLECC_VER_MINOR__ 0
#endif
#ifndef __COMPATIBLECC_VER_BUILD__
#define __COMPATIBLECC_VER_BUILD__ 0
#endif

// Device-only, like __CUDA_ARCH__: undefined during the host pass.
#if defined(__HGGC_ARCH__) && !defined(COMPATIBLE_ARCH)
#if __HGGC_ARCH__ == 100
#define COMPATIBLE_ARCH 800
#elif __HGGC_ARCH__ == 150
#define COMPATIBLE_ARCH 890
#else
#define COMPATIBLE_ARCH __HGGC_ARCH__
#endif
#endif

// hgcc defines this only for device code, like __CUDA_ARCH_LIST__.
#ifndef COMPATIBLE_ARCH_LIST
#if defined(__HGGCCC__)
#if defined(COMPATIBLE_ARCH)
#define COMPATIBLE_ARCH_LIST COMPATIBLE_ARCH
#else
#define COMPATIBLE_ARCH_LIST 800
#endif
#endif
#endif

// Un-sailified Torch wheels key their device branches on the NVIDIA macro
// spellings, which for them the CUDA-compatible nvcc driver supplies. Every
// translation unit here is compiled by the hgcc driver, which defines neither,
// so supply both. __CUDACC__ holds in either pass; __CUDA_ARCH__ only where
// __HGGC_ARCH__ marks the device pass, mapped like COMPATIBLE_ARCH above.
#if defined(__HGGCCC__) && !defined(__CUDACC__)
#define __CUDACC__ 1
#define __CUDACC_VER_MAJOR__ __COMPATIBLECC_VER_MAJOR__
#define __CUDACC_VER_MINOR__ __COMPATIBLECC_VER_MINOR__
#define __CUDACC_VER_BUILD__ __COMPATIBLECC_VER_BUILD__
#endif
#if defined(__HGGC_ARCH__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ COMPATIBLE_ARCH
#endif
