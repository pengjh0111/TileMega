// SPDX-License-Identifier: BSD-3-Clause
// R8 A-a: run each rewritten TaskBody on fixture inputs and write what it
// produced, so it can be compared element by element against the PyTorch
// operator that generated those inputs. The bodies are called through their
// own entry points -- no reimplementation of their arithmetic here.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/QKNormTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>

using namespace tilemega::codegen;
using Arch = ::tilemega::arch::Sm89;
static constexpr int kThreads = 128;

/// The members the three bodies touch, laid out like the harness union.
union CheckSmem {
  float attention[TILEMEGA_ATTENTION_SCRATCH_EXTENT + kThreads / 32];
  float rms[kThreads];
};

__global__ void RmsNormCheck(ModelElement const* in, ModelElement const* w,
                             ModelElement* out, int tokens, int hidden) {
  __shared__ CheckSmem smem;
  for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
    RMSNormTaskBody<Arch, CheckSmem, kThreads>::RunRow(
        in + static_cast<std::size_t>(token) * hidden, w,
        out + static_cast<std::size_t>(token) * hidden, hidden, smem.rms);
    __syncthreads();
  }
}

/// QK-norm is one head of the same row body, which is how the harness reaches
/// it; `tasks` is the flattened (token, head).
__global__ void QkNormCheck(ModelElement const* in, ModelElement const* w,
                            ModelElement* out, int tasks, int head_dim) {
  __shared__ CheckSmem smem;
  for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
    RMSNormTaskBody<Arch, CheckSmem, kThreads>::RunRow(
        in + static_cast<std::size_t>(task) * head_dim, w,
        out + static_cast<std::size_t>(task) * head_dim, head_dim, smem.rms);
    __syncthreads();
  }
}

__global__ void AttentionCheck(Params p, StageDesc stage, int queries) {
  __shared__ CheckSmem smem;
  for (int query = blockIdx.x; query < queries; query += gridDim.x) {
    AttentionTaskBody<Arch, CheckSmem, kThreads>::RunTask(p, stage, smem, query);
    __syncthreads();
  }
}

static std::vector<char> Read(std::string const& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) { std::fprintf(stderr, "missing %s\n", path.c_str()); std::exit(2); }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static void Write(std::string const& path, void const* data, std::size_t bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<char const*>(data), static_cast<std::streamsize>(bytes));
}

template <class T>
static T* Upload(std::vector<char> const& host) {
  void* device = nullptr;
  cudaMalloc(&device, host.size());
  cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice);
  return static_cast<T*>(device);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: operator_check <fixture dir>\n"); return 2; }
  std::string dir = argv[1];
  auto meta = Read(dir + "/shapes.txt");
  int tokens = 0, hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, past = 0;
  std::sscanf(meta.data(), "%d %d %d %d %d %d", &tokens, &hidden, &heads,
              &kv_heads, &head_dim, &past);
  int const total = past + tokens;

  { // RMSNorm
    auto* in = Upload<ModelElement>(Read(dir + "/rmsnorm_in.bin"));
    auto* w = Upload<ModelElement>(Read(dir + "/rmsnorm_w.bin"));
    ModelElement* out = nullptr;
    cudaMalloc(&out, sizeof(ModelElement) * tokens * hidden);
    RmsNormCheck<<<tokens, kThreads>>>(in, w, out, tokens, hidden);
    cudaDeviceSynchronize();
    std::vector<ModelElement> host(static_cast<std::size_t>(tokens) * hidden);
    cudaMemcpy(host.data(), out, host.size() * sizeof(ModelElement), cudaMemcpyDeviceToHost);
    Write(dir + "/rmsnorm_device.bin", host.data(), host.size() * sizeof(ModelElement));
  }
  { // QK-norm, one row per (token, head)
    auto* in = Upload<ModelElement>(Read(dir + "/qknorm_in.bin"));
    auto* w = Upload<ModelElement>(Read(dir + "/qknorm_w.bin"));
    int const tasks = tokens * heads;
    ModelElement* out = nullptr;
    cudaMalloc(&out, sizeof(ModelElement) * tasks * head_dim);
    QkNormCheck<<<tasks, kThreads>>>(in, w, out, tasks, head_dim);
    cudaDeviceSynchronize();
    std::vector<ModelElement> host(static_cast<std::size_t>(tasks) * head_dim);
    cudaMemcpy(host.data(), out, host.size() * sizeof(ModelElement), cudaMemcpyDeviceToHost);
    Write(dir + "/qknorm_device.bin", host.data(), host.size() * sizeof(ModelElement));
  }
  { // Attention: the rewritten softmax, through the body's own RunTask
    auto* q = Upload<ModelElement>(Read(dir + "/attn_q.bin"));
    auto* k = Upload<ModelElement>(Read(dir + "/attn_k.bin"));
    auto* v = Upload<ModelElement>(Read(dir + "/attn_v.bin"));
    ModelElement* context = nullptr;
    std::size_t const context_elements =
        static_cast<std::size_t>(tokens) * heads * head_dim;
    cudaMalloc(&context, sizeof(ModelElement) * context_elements);
    ModelElement* table[4] = {q, k, v, context};
    ModelElement** device_table = nullptr;
    cudaMalloc(&device_table, sizeof(table));
    cudaMemcpy(device_table, table, sizeof(table), cudaMemcpyHostToDevice);
    Params p{};
    p.dims.seq = tokens;
    p.dims.past = past;
    p.dims.total = total;
    p.buffers = device_table;
    StageDesc stage{};
    stage.kind = TaskKind::kAttention;
    stage.extent = static_cast<std::uint32_t>(heads);
    stage.width = static_cast<std::uint32_t>(head_dim);
    stage.group = static_cast<std::uint32_t>(heads / kv_heads);
    stage.operand[0] = 0; stage.operand[1] = 1;
    stage.operand[2] = 2; stage.operand[3] = 3;
    stage.operand[6] = 0; stage.operand[7] = kNoOperand;
    AttentionCheck<<<tokens * heads, kThreads>>>(p, stage, tokens * heads);
    cudaDeviceSynchronize();
    std::vector<ModelElement> host(context_elements);
    cudaMemcpy(host.data(), context, host.size() * sizeof(ModelElement), cudaMemcpyDeviceToHost);
    Write(dir + "/attention_device.bin", host.data(), host.size() * sizeof(ModelElement));
  }
  cudaError_t const status = cudaGetLastError();
  std::printf("OPERATOR_CHECK status=%s\n", cudaGetErrorString(status));
  return status == cudaSuccess ? 0 : 1;
}
