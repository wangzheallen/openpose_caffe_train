// Note: 
// This file is based in data_layer.hpp, deleted in `Switched multi-GPU to NCCL` (Jan 6, 2017)
// https://github.com/BVLC/caffe/commits/master?after=4efdf7ee49cffefdd7ea099c00dc5ea327640f04+156

#ifndef CAFFE_OPENPOSE_CPM_DATA_LAYER_HPP
#define CAFFE_OPENPOSE_CPM_DATA_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/openpose/data_reader.hpp"
#include "caffe/openpose/cpm_data_transformer.hpp"

namespace caffe {

template <typename Dtype>
class CPMDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CPMDataLayer(const LayerParameter& param);
  virtual ~CPMDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // CPMDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "CPMData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
  // OpenPose: different than data_layer.hpp
  Blob<Dtype> transformed_label_; // add another blob
  CPMTransformationParameter cpm_transform_param_;
  shared_ptr<CPMDataTransformer<Dtype> > cpm_data_transformer_;
  // OpenPose: different than data_layer.hpp end
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
