#ifndef __CUDA_DEVICE_VECTOR__HPP__
#define __CUDA_DEVICE_VECTOR__HPP__

#include <cuda_runtime.h>
#include <opencv2/core/cuda/common.hpp>

template <class T>
class CudaDeviceVector {
  public:
    class Dev {
      private:
        Dev() {
        }

      public:

#ifdef __CUDACC__
        __forceinline__ __device__ void push(const T& v) {
          int idx = atomicAdd(size_ptr_, 1);
          if (idx < max_size_) {
            ptr_[idx] = v;
          }
        }

        __forceinline__ __device__ int size() const {
          return *size_ptr_;
        }

        __forceinline__ __device__ const T& operator[](int i) const {
          return ptr_[i];
        }
#endif

      private:
        int max_size_;
        int* size_ptr_;
        T* ptr_;

        friend class CudaDeviceVector;
    };

  public:
    __host__ CudaDeviceVector(int max_size) {
      dev_.max_size_ = max_size;
      cudaSafeCall(cudaMalloc(&dev_.ptr_, sizeof(T)*max_size));
      cudaSafeCall(cudaMalloc(&dev_.size_ptr_, sizeof(int)));

      clear();
    }

    ~CudaDeviceVector() {
      cudaSafeCall(cudaFree(dev_.ptr_));
      cudaSafeCall(cudaFree(dev_.size_ptr_));
    }

    CudaDeviceVector(const CudaDeviceVector&) = delete;
    CudaDeviceVector& operator = (const CudaDeviceVector&) = delete;

    operator Dev() {
      return dev_;
    }

    operator const Dev() const {
      return dev_;
    }

    __host__ void clear() {
      cudaSafeCall(cudaMemset(dev_.size_ptr_, 0, sizeof(int)));
    }

    __host__ int size() const {
      int size;
      cudaSafeCall(cudaMemcpy(
            &size, dev_.size_ptr_, sizeof(int), 
            cudaMemcpyDeviceToHost));
      return std::min(size, dev_.max_size_);
    }

    __host__ void download(std::vector<T>& dst) const {
      dst.resize(size());
      cudaSafeCall(cudaMemcpy(
            dst.data(), dev_.ptr_, sizeof(T)*dst.size(),
            cudaMemcpyDeviceToHost));
    }

    __host__ void upload(const std::vector<T>& src) {
      upload_size_ = src.size();
      cudaSafeCall(cudaMemcpy(
            dev_.ptr_, src.data(), sizeof(T)*src.size(),
            cudaMemcpyHostToDevice));
      cudaSafeCall(cudaMemcpy(
            dev_.size_ptr_, &upload_size_, sizeof(int),
            cudaMemcpyHostToDevice));
    }


  private:
    Dev dev_;
    mutable int upload_size_;
};


#endif
