#ifndef SHARED_CONV_BLOBS_HPP
#define SHARED_CONV_BLOBS_HPP

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
class SharedConvBlobs {

public:

  SharedConvBlobs() {
    col_buffer_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    bias_multiplier_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  }

  shared_ptr<Blob<Dtype> > getColBufferBlob() {
    return col_buffer_;
  }

  shared_ptr<Blob<Dtype> > getBiasMultiplierBlob() {
    return bias_multiplier_;
  }

private:

  shared_ptr<Blob<Dtype> > col_buffer_;
  shared_ptr<Blob<Dtype> > bias_multiplier_;
};
 
}

#endif