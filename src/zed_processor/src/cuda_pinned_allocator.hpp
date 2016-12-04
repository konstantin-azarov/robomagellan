#ifndef __CUDA_PINNED_ALLOCATOR__HPP__
#define __CUDA_PINNED_ALLOCATOR__HPP__

#include <cuda_runtime.h>
#include <opencv2/core/cuda/common.hpp>

template<typename T>
class CudaPinnedAllocator: public std::allocator<T>
{
  typedef std::allocator<T> parent;

public:
  using typename parent::pointer;
  using typename parent::size_type;
  
  pointer allocate(size_type n, const void * = 0)
  {
    pointer p = new T[n];
  //  cudaSafeCall(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault));

    return p;
  }

  void deallocate(pointer p, size_type)
  {
    delete[] p;
//    cudaSafeCall(cudaFreeHost(p));
  }

  template<typename O>
  struct rebind
  {
    typedef CudaPinnedAllocator<O> other;
  };
};

template<typename T1, typename T2>
inline bool operator == (const CudaPinnedAllocator<T1> &,
                       const CudaPinnedAllocator<T2> &) {
  return true;
}

template<typename T1, typename T2>
inline bool operator != (const CudaPinnedAllocator<T1> &,
                       const CudaPinnedAllocator<T2> &) {
  return false;
}

template <class T>
using PinnedVector = std::vector<T, CudaPinnedAllocator<T>>;

#endif
