// SPDX-License-Identifier: Apache-2.0
#pragma once

// Stock torch still includes this CUDA spelling under __CUDACC__. Include
// guards alone cannot satisfy the file lookup on the native HG include path.
// Use the SDK's native scalars and aliases without a CUDA header dependency.
#include "ppu_compat.h"
