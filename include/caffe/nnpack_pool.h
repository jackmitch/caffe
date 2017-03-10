#pragma once

#include <boost/noncopyable.hpp>

#ifdef _MSC_VER
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "nnpack.h"

namespace caffe {
class NNPACKPool : public boost::noncopyable {
 public:
   NNPACKPool() {

   int num_mkl_threads;

#ifdef _MSC_VER
   SYSTEM_INFO sysinfo;
   GetSystemInfo(&sysinfo);
   num_mkl_threads = sysinfo.dwNumberOfProcessors;
#else
   num_mkl_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif

     if (num_mkl_threads > 1) {
       pool_ = pthreadpool_create(num_mkl_threads);
     } else {
       pool_ = NULL;
     }

   }
  ~NNPACKPool() {
    if (pool_) {
      pthreadpool_destroy(pool_);
    }
    pool_ = NULL;
  }

  pthreadpool_t pool() { return pool_; };

 private:
  pthreadpool_t pool_;
};

}
