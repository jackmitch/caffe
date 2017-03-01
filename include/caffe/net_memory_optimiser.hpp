#ifndef CAFFE_NET_MEMORY_HPP_
#define CAFFE_NET_MEMORY_HPP_

#include "caffe/net.hpp"

namespace caffe {

/**
 * @brief Finds which layers can share the activations by counting which activations are in use
 *        at the same time. A layer takes a syncedmem out of the avalible pool and puts it back once 
 *        it has been used by all layers that require it as an input. If the pool is empty when a
 *        layer tries to take from the pool a new SyncedMem is added to the pool.
 *        The blobs are then made to share the SyncedMem they took fro the pool
 */
template <typename Dtype>
class NetMemoryOptimiser {

  class SyncedMemoryPool {
  public:
    SyncedMemoryPool();
    uint64_t byteSize() const;
    void clear();
    int take(int idx = -1);
    void give(int idx);
    shared_ptr<SyncedMemory> get(int idx);

  private:
    std::vector<shared_ptr<SyncedMemory> > pool_;
    std::vector<int> in_use_count_;
  };

public:
  NetMemoryOptimiser(Net<Dtype>& net);

  void optimise();

private:
  void buildShareMap_();
  void buildExcludeList_();
  void buildSyncedMemPool_();
  void assignMemory_();

  void excludeBlob(const string& blob_name);
  bool isExcluded(const string& blob_name);

  string getOrRegisterBlobShare(const string& name, const string& share_name = "");

// atrributes
private:
  Net<Dtype>& net_;

  uint64_t raw_byte_count_;
  uint64_t optimised_byte_count_;

  map<string, string> blob_map_; // blob_name, blob_name_it_shares
  set<string> excluded_blobs_;
  SyncedMemoryPool syncedmem_pool_;
  map<string, int> pool_idx_;

  DISABLE_COPY_AND_ASSIGN(NetMemoryOptimiser);
};

}
#endif