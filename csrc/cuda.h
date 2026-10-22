// SPDX-License-Identifier: Apache-2.0
#pragma once

// Stock torch's ATen/Dispatch.h includes cuda.h for CUDA_VERSION under
// __CUDACC__. HG compilation uses the native SDK, so expose only that version
// gate here; this header does not provide the CUDA driver API.
#include "ppu_compat.h"

#ifndef CUDA_VERSION
#define CUDA_VERSION COMPATIBLE_VERSION
#endif
