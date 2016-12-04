#include <cuda_runtime.h>
#include <sys/mman.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <chrono>

#include "benchmark/benchmark.h"


int readMemory(uint32_t* mem, int length) {
  int i = 0;
  int sum = 0;
  while (length--) {
    sum += mem[i++];
  }

  return sum;
}

void writeMemory(uint32_t* mem, int length) {
  int i = 0;
  int v = 0;
  while (length--) {
    mem[i++] = v++;
  }
}

const int kItems = 10000000;
const int kMemSize = kItems * sizeof(int);

void fillMemory(uint32_t* p) {
  for (int i = 0; i < kItems; ++i) {
    p[i] = 1;
  }
}

#define check(x) { if ((x) != kItems) state.SkipWithError("Validation fail"); }

static void BM_malloc_read(benchmark::State& state) {
  uint32_t* p = (uint32_t*)malloc(kMemSize);
  fillMemory(p);
  while(state.KeepRunning())
    check(readMemory(p, kItems));
  free(p);

  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_malloc_write(benchmark::State& state) {
  uint32_t* p = (uint32_t*)malloc(kMemSize);
  while(state.KeepRunning())
    writeMemory(p, kItems);
  free(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}

static void BM_mmap_read(benchmark::State& state) {
  uint32_t* p = (uint32_t*)mmap(
      0, kMemSize, 
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_LOCKED | MAP_ANONYMOUS,
      -1, 0);
  fillMemory(p);

  while(state.KeepRunning())
    check(readMemory(p, kItems));

  munmap(p, kMemSize);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_mmap_write(benchmark::State& state) {
  uint32_t* p = (uint32_t*)mmap(
      0, kMemSize, 
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_LOCKED | MAP_ANONYMOUS,
      -1, 0);

  while(state.KeepRunning())
    writeMemory(p, kItems);

  munmap(p, kMemSize);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}

#define cudaSafeCall(x) if ((x)) { \
  state.SkipWithError(cudaGetErrorString(cudaGetLastError())); \
  return; \
}

static void BM_cuda_copy_h2d(benchmark::State& state) {
  uint32_t* host_p = (uint32_t*)malloc(kMemSize);
  uint32_t* dev_p;
  cudaSafeCall(cudaMalloc(&dev_p, kMemSize));
  fillMemory(host_p);

  while (state.KeepRunning()) {
    cudaMemcpy(dev_p, host_p, kMemSize, cudaMemcpyHostToDevice);
  }

  cudaFree(dev_p);
  free(host_p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}

static void BM_cuda_copy_d2h(benchmark::State& state) {
  uint32_t* host_p = (uint32_t*)malloc(kMemSize);
  uint32_t* dev_p;
  cudaSafeCall(cudaMalloc(&dev_p, kMemSize));
  fillMemory(host_p);
  cudaMemcpy(dev_p, host_p, kMemSize, cudaMemcpyHostToDevice);

  while (state.KeepRunning()) {
    cudaMemcpy(host_p, dev_p, kMemSize, cudaMemcpyDeviceToHost);
  }

  cudaFree(dev_p);
  free(host_p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}

static void BM_cuda_copy_h2d_async(benchmark::State& state) {
  uint32_t* host_p = (uint32_t*)malloc(kMemSize);
  uint32_t* dev_p;
  cudaSafeCall(cudaMalloc(&dev_p, kMemSize));
  fillMemory(host_p);

  while (state.KeepRunning()) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(dev_p, host_p, kMemSize, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    state.SetIterationTime(
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
  }

  cudaFree(dev_p);
  free(host_p);
}

static void BM_cuda_copy_d2h_async(benchmark::State& state) {
  uint32_t* host_p = (uint32_t*)malloc(kMemSize);
  uint32_t* dev_p;
  cudaSafeCall(cudaMalloc(&dev_p, kMemSize));
  fillMemory(host_p);
  cudaMemcpy(dev_p, host_p, kMemSize, cudaMemcpyHostToDevice);

  while (state.KeepRunning()) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(host_p, dev_p, kMemSize, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    state.SetIterationTime(
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
  }

  cudaFree(dev_p);
  free(host_p);
}



static void BM_cuda_malloc_read(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaMallocHost(&p, kMemSize));
  fillMemory(p);

  while(state.KeepRunning())
    check(readMemory(p, kItems));

  cudaFreeHost(p);

  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_cuda_malloc_write(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaMallocHost(&p, kMemSize));

  while(state.KeepRunning())
    writeMemory(p, kItems);

  cudaFreeHost(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}

static void BM_pinned_read(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaHostAlloc(&p, kMemSize, cudaHostAllocDefault));
  fillMemory(p);

  while(state.KeepRunning())
    check(readMemory(p, kItems));

  cudaFreeHost(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_pinned_write(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaHostAlloc(&p, kMemSize, cudaHostAllocDefault));

  while(state.KeepRunning())
    writeMemory(p, kItems);

  cudaFreeHost(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_mapped_read(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaHostAlloc(&p, kMemSize, cudaHostAllocMapped));
  fillMemory(p);

  while(state.KeepRunning())
    check(readMemory(p, kItems));

  cudaFreeHost(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_mapped_write(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaHostAlloc(&p, kMemSize, cudaHostAllocMapped));

  while(state.KeepRunning())
    writeMemory(p, kItems);

  cudaFreeHost(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_managed_read(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaMallocManaged(&p, kMemSize));
  fillMemory(p);

  while(state.KeepRunning())
    check(readMemory(p, kItems));

  cudaFree(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}


static void BM_managed_write(benchmark::State& state) {
  uint32_t* p;
  cudaSafeCall(cudaMallocManaged(&p, kMemSize));

  while(state.KeepRunning())
    writeMemory(p, kItems);

  cudaFree(p);
  
  state.SetBytesProcessed(state.iterations() * kMemSize);
}

BENCHMARK(BM_malloc_read);
BENCHMARK(BM_malloc_write);
/* BENCHMARK(BM_mmap_read); */
/* BENCHMARK(BM_mmap_write); */
BENCHMARK(BM_cuda_copy_h2d);
BENCHMARK(BM_cuda_copy_d2h);
BENCHMARK(BM_cuda_copy_h2d_async)->UseManualTime();
BENCHMARK(BM_cuda_copy_d2h_async)->UseManualTime();
BENCHMARK(BM_cuda_malloc_read);
BENCHMARK(BM_cuda_malloc_write);
BENCHMARK(BM_pinned_read);
BENCHMARK(BM_pinned_write);
BENCHMARK(BM_mapped_read);
BENCHMARK(BM_mapped_write);
BENCHMARK(BM_managed_read);
BENCHMARK(BM_managed_write);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
