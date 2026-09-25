// SPDX-License-Identifier: BSD-3-Clause
// Stable, C-compatible interface between generated serving plans and Python.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { TM_SERVING_ABI_VERSION = 1 };
enum { TM_SERVING_PREFILL = 0, TM_SERVING_DECODE = 1 };
enum { TM_SERVING_L1 = 1, TM_SERVING_L2 = 2 };
enum { TM_BUFFER_INTERNAL = 0, TM_BUFFER_EXTERNAL = 1 };
enum { TM_DTYPE_BF16 = 0, TM_DTYPE_F32 = 1, TM_DTYPE_I32 = 2 };

typedef struct {
  uint32_t abi_version, phase;
  int32_t batch_lo, batch_hi, seq, past_lo, past_hi, capacity;
  int32_t grid, residency;
  uint32_t modes;
  uint32_t buffer_count;
} tm_plan_info;

typedef struct {
  const char* name;
  uint32_t role;
  uint32_t dtype;
  uint64_t elements_constant, elements_per_batch;
  const char* pack_json;
} tm_buffer_info;

int tm_plan_query(tm_plan_info* out);
int tm_plan_buffer(uint32_t index, tm_buffer_info* out);
void* tm_plan_create(int batch, void* const* external, int device);
int tm_plan_set_steps(void* plan, const int32_t* past, uint32_t count);
// iteration is the number of previous launches of this mode on this plan.
// L1 and L2 each advance independently; event counters are never reset.
int tm_plan_launch(void* plan, uint32_t step, uint32_t mode,
                   uint64_t iteration, void* stream);
void tm_plan_destroy(void* plan);

#ifdef __cplusplus
}
#endif
