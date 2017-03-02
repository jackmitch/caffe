#ifndef SHARED_CONV_BLOBS_HPP
#define SHARED_CONV_BLOBS_HPP

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
class SharedConvBlobs {

public:

  static void assign() {
    col_buffer_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    bias_multiplier_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  }

  static void unassign() {
    col_buffer_.reset();
    bias_multiplier_.reset();
  }

  static shared_ptr<Blob<Dtype> > getColBufferBlob() {
    return col_buffer_;
  }

  static shared_ptr<Blob<Dtype> > getBiasMultiplierBlob() {
    return bias_multiplier_;
  }

private:

  static shared_ptr<Blob<Dtype> > col_buffer_;
  static shared_ptr<Blob<Dtype> > bias_multiplier_;
};
 
}

#endif