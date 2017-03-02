#ifndef SHARED_CUDNN_DATA_HPP
#define SHARED_CUDNN_DATA_HPP

#include "caffe/blob.hpp"

namespace caffe {
#ifdef USE_CUDNN
  template <typename Dtype>
  class SharedCuDNNData {

  public:

    static void restart() {
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    static void* resizeConvWorkspace(size_t size) {
      cudaFree(workspaceData);
      cudaError_t err = cudaMalloc(&workspaceData, workspaceSizeInBytes);
      if(err == cudaSuccess) {
        return workspaceData;
      }
      return NULL;
    }

    static void cleanUp(void *data) {
      cudaFree(data);
    }

    static size_t getConvWorkspaceSize() const { return workspaceSizeInBytes; }

    static void* getConvWorkspaceData() const { return workspaceData; }

  private:

    static size_t workspaceSizeInBytes;  // size of underlying storage
    static void *workspaceData;
  };
#endif
}

#endif