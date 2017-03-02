#ifndef SHARED_CUDNN_DATA_HPP
#define SHARED_CUDNN_DATA_HPP

#include "caffe/blob.hpp"

namespace caffe {
#ifdef USE_CUDNN
  template <typename Dtype>
  class SharedCuDNNData {

  public:

    SharedCuDNNData() {
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    inline ~SharedCuDNNData() {
      if(workspaceData) {
        cudaFree(workspaceData);
      }
    }

    inline void* resizeConvWorkspace(size_t size) {
      cudaFree(workspaceData);
      workspaceSizeInBytes = size;
      cudaError_t err = cudaMalloc(&workspaceData, workspaceSizeInBytes);
      if(err == cudaSuccess) {
        return workspaceData;
      }
      return NULL;
    }

    inline size_t getConvWorkspaceSize() { return workspaceSizeInBytes; }

    inline void* getConvWorkspaceData() { return workspaceData; }

  private:

    size_t workspaceSizeInBytes;  // size of underlying storage
    void *workspaceData;
  };
#else
  template <typename Dtype>
  class SharedCuDNNData {

  };
#endif
}

#endif