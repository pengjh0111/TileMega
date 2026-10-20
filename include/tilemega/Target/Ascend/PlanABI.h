// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace tilemega::ascend::abi {
inline constexpr uint32_t kVersion=1;
// Same task/dependency/wait meanings as the original TaskRef. Architecture
// metadata lives beside it, not in its logical coordinates or trigger count.
struct TaskRef {
  uint32_t stage,logical_task,dependency_begin,dependency_count,wait_begin,wait_count,flags;
};
struct TaskMeta { uint32_t owner,event; };
struct Metadata {
  uint32_t version,groups,workers,tasks,dependencies,epochs,stages,buffers;
};
struct Stage {
  uint32_t kind,a,b,c,rows,columns,k,tasks;
  float epsilon;
};
static_assert(sizeof(TaskRef)==28 && alignof(TaskRef)==4);
static_assert(offsetof(TaskRef,logical_task)==4 && offsetof(TaskRef,wait_begin)==16);
static_assert(sizeof(TaskMeta)==8 && sizeof(Metadata)==32);
static_assert(sizeof(Stage)==36 && alignof(Stage)==4 && offsetof(Stage,epsilon)==32);
static_assert(std::is_standard_layout_v<Stage> && std::is_trivially_copyable_v<Stage>);
static_assert(std::is_standard_layout_v<TaskRef> && std::is_trivially_copyable_v<TaskRef>);
static_assert(std::is_standard_layout_v<Metadata> && std::is_trivially_copyable_v<Metadata>);
} // namespace tilemega::ascend::abi
