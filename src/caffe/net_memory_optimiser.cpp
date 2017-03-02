
#include <map>
#include <set>
#include <vector>
#include <string>
#include "caffe/net_memory_optimiser.hpp"

namespace caffe {

template <typename Dtype>
NetMemoryOptimiser<Dtype>::SyncedMemoryPool::SyncedMemoryPool()
{

}

template <typename Dtype>
uint64_t NetMemoryOptimiser<Dtype>::SyncedMemoryPool::byteSize() const {
  uint64_t size = 0;
  for (size_t n = 0; n < pool_.size(); n++) {
    size += pool_[n]->size();
  }
  return size;
}

template <typename Dtype>
shared_ptr<SyncedMemory> NetMemoryOptimiser<Dtype>::SyncedMemoryPool::get(int idx) {
  return pool_[idx];
}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::SyncedMemoryPool::clear() {
  in_use_count_.clear();
  pool_.clear();
}

template <typename Dtype>
int NetMemoryOptimiser<Dtype>::SyncedMemoryPool::take(int idx)
{
  if (idx >= 0) {
    in_use_count_[idx]++;
    return idx;
  }

  for (size_t n = 0; n < pool_.size(); n++) {
    if (in_use_count_[n] == 0) {
      in_use_count_[n]++;
      return n;
    }
  }

  pool_.push_back(shared_ptr<SyncedMemory>(new SyncedMemory(1)));
  in_use_count_.push_back(1);
  return pool_.size() - 1;
}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::SyncedMemoryPool::give(int idx)
{
  in_use_count_[idx]--;
  CHECK_GE(in_use_count_[idx], 0);
}

template <typename Dtype>
NetMemoryOptimiser<Dtype>::NetMemoryOptimiser(Net<Dtype>& net, const MemoryOptimisationParams& params)
  :
net_(net),
params_(params)
{

}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::excludeBlob(const string& blob_name) {

  LOG(INFO) << "excluding " << blob_name;
  excluded_blobs_.insert(getOrRegisterBlobShare(blob_name));
}

template <typename Dtype>
string NetMemoryOptimiser<Dtype>::getOrRegisterBlobShare(const string& name, const string& share_name) {

  if (share_name.empty() && blob_map_.find(name) != blob_map_.end()){
    return blob_map_[name];
  }
  else{
    blob_map_[name] = share_name.empty() ? name : share_name;
    return name;
  }
}

template <typename Dtype>
bool NetMemoryOptimiser<Dtype>::isExcluded(const string& blob_name) {
  return excluded_blobs_.find(blob_name) != excluded_blobs_.end();
}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::buildShareMap_() {
  // for each output blob store its root, e.g. a split layer just shares its input
  // to a number of output blobs so all output blobs need to reference the input blob
  blob_map_.clear();

  for (int i = 0; i < net_.layers_.size(); ++i) {

    const vector<Blob<Dtype>* >& layer_outputs = net_.top_vecs_[i];
    const vector<Blob<Dtype>* >& layer_inputs = net_.bottom_vecs_[i];

    for (int t = 0; t < layer_outputs.size(); ++t) {

      const string& output_name = net_.blob_names_[net_.top_id_vecs_[i][t]];

      string output_share_name = getOrRegisterBlobShare(output_name);

      for (int b = 0; b < layer_inputs.size(); ++b) {

        const string& input_name = net_.blob_names_[net_.bottom_id_vecs_[i][b]];

        string input_share_name = getOrRegisterBlobShare(input_name);

        // These layers share input and output blobs even if the blob names are different
        if (string("Reshape") == net_.layers_[i]->type() && t == 0 && b == 0) {
          getOrRegisterBlobShare(output_share_name, input_share_name);
        }
        if (string("Flatten") == net_.layers_[i]->type() && t == 0 && b == 0) {
          getOrRegisterBlobShare(output_share_name, input_share_name);
        }
        if (string("Split") == net_.layers_[i]->type()) {
          getOrRegisterBlobShare(output_share_name, input_share_name);
        }
      
      } // all layer outputs
    } // all layer inputs
  } // all layers
}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::buildExcludeList_()
{
  excluded_blobs_.clear();

  // exclude all blobs specified in the params
  for (int i = 0; i < params_.excluded_layer_size(); ++i) {
    for (int j = 0; j < net_.layers_.size(); ++j) {
      if (net_.layers_[j]->layer_param().name() == params_.excluded_layer(i))
      {
        // ignore all top and bottom layers
        for (int t = 0; t < net_.top_vecs_[j].size(); ++t) {
          excludeBlob(net_.blob_names_[net_.top_id_vecs_[j][t]]);
        }
        for (int b = 0; b < net_.top_vecs_[j].size(); ++b) {
          excludeBlob(net_.blob_names_[net_.bottom_id_vecs_[j][b]]);
        }
      }
    }
  }

  // Exclude input layers as they might be preloaded by different threads
  for (int i = 0; i < net_.net_input_blob_indices_.size(); ++i) {
    excludeBlob(net_.blob_names_[net_.net_input_blob_indices_[i]]);
  }

  // Exclude data layers
  for (int i = 0; i < net_.layers_.size(); ++i) {
    if (net_.bottom_vecs_[i].size() == 0) {
      for (int t = 0; t < net_.top_vecs_[i].size(); ++t) {
        excludeBlob(net_.blob_names_[net_.top_id_vecs_[i][t]]);
      }
    }
  }
/*
  // exclude output layers as they might needed externally 
  for (int i = 0; i < net_.net_output_blob_indices_.size(); ++i) {
    excludeBlob(net_.blob_names_[net_.net_output_blob_indices_[i]]);
  }

  // Exclude all losses
  for (int i = 0; i < net_.layers_.size(); ++i) {
    for (int t = 0; t < net_.top_vecs_[i].size(); ++t) {
      if (net_.layers_[i]->loss(t)) {
        excludeBlob(net_.blob_names_[net_.top_id_vecs_[i][t]]);
      }
    }
  }
*/
}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::buildSyncedMemPool_() {

  syncedmem_pool_.clear();
  pool_idx_.clear();

  for (int i = 0; i < net_.layers_.size(); ++i) {

    // get something form the pool for each layer output
    for (int t = 0; t < net_.top_vecs_[i].size(); ++t) {

      const string& blob_name = net_.blob_names_[net_.top_id_vecs_[i][t]];
      const string& share_name = blob_map_[blob_name];

      if (isExcluded(share_name))
        continue;

      if (pool_idx_.find(share_name) != pool_idx_.end()) {
        syncedmem_pool_.take(pool_idx_[share_name]);
      }
      else {
        int pidx = syncedmem_pool_.take();
        pool_idx_[share_name] = pidx;
      }
    }

    // put back into the pool for each layer input that is no longer needed
    for (int b = 0; b < net_.bottom_vecs_[i].size(); ++b) {

      const string& blob_name = net_.blob_names_[net_.bottom_id_vecs_[i][b]];
      const string& share_name = blob_map_[blob_name];

      if (isExcluded(share_name))
        continue;

      syncedmem_pool_.give(pool_idx_[share_name]);
    }
  }
}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::assignMemory_() {

  raw_byte_count_ = 0;
  optimised_byte_count_ = 0;

  for (int i = 0; i < net_.blobs_.size(); ++i) {

    const string& blob_name = net_.blob_names_[i];
    const string& share_name = blob_map_[blob_name];

    const size_t bytes = net_.blobs_[i]->count() * sizeof(Dtype);

    raw_byte_count_ += bytes;

    if (pool_idx_.find(share_name) != pool_idx_.end()) {
      shared_ptr<SyncedMemory> syncedmem = syncedmem_pool_.get(pool_idx_[share_name]);
      net_.blobs_[i]->ShareData(syncedmem);
      syncedmem->resize(bytes);
    }
    else {
      optimised_byte_count_ += bytes;
    }
  }
  
  optimised_byte_count_ += syncedmem_pool_.byteSize();

  LOG(INFO) << "raw memory " << raw_byte_count_ / 1e6 << "MB opt memory "
    << optimised_byte_count_ / 1e6 << "MB " << " compressed : " 
    << 100 * optimised_byte_count_ / raw_byte_count_ << "% of original";
}

template <typename Dtype>
void NetMemoryOptimiser<Dtype>::optimise() {

  if (params_.optimise()) {
    CHECK_EQ(net_.phase(), TEST);

    // build a map of which blobs are shared with other blobs in the net
    buildShareMap_();

    // exclude some blobs from sharing
    buildExcludeList_();

    // build the memory pool required to serve the blobs
    buildSyncedMemPool_();

    // assign the shared memory to the blobs
    assignMemory_();
  }
}

INSTANTIATE_CLASS(NetMemoryOptimiser);

} // caffe namespace