// Mobili

#ifndef LANEDET_H
#define LANEDET_H

#include <opencv2/core/core.hpp>

#include <net.h>
#include "android_log.h"

// https://github.com/IrohXu/lanenet-lane-detection-pytorch
class LaneDet {
 public:
  LaneDet();

  int load(AAssetManager* mgr, bool use_gpu = false);

  int detect_and_draw(const cv::Mat& rgb);
  //
  // int draw(cv::Mat& rgb, const std::vector<Object>& objects);

 private:
  bool loaded_ = false;
  ncnn::Net lanedet;
  ncnn::UnlockedPoolAllocator blob_pool_allocator;
  ncnn::PoolAllocator workspace_pool_allocator;
};

#endif  // LANEDET_H
