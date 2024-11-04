// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "lanedet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

LaneDet::LaneDet() {
  blob_pool_allocator.set_size_compare_ratio(0.f);
  workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int LaneDet::load(AAssetManager* mgr, bool use_gpu) {
  loaded_ = true;
  lanedet.clear();
  blob_pool_allocator.clear();
  workspace_pool_allocator.clear();

  ncnn::set_cpu_powersave(2);
  ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

  lanedet.opt = ncnn::Option();

#if NCNN_VULKAN
  lanedet.opt.use_vulkan_compute = use_gpu;
#endif

  lanedet.opt.num_threads = ncnn::get_big_cpu_count();
  lanedet.opt.blob_allocator = &blob_pool_allocator;
  lanedet.opt.workspace_allocator = &workspace_pool_allocator;

  char parampath[256];
  char modelpath[256];
  sprintf(parampath, "ncnn_model.ncnn.param");
  sprintf(modelpath, "ncnn_model.ncnn.bin");
  LOGI("load lanedet model: %s and %s", parampath, modelpath);

  lanedet.load_param(mgr, parampath);
  lanedet.load_model(mgr, modelpath);

  return 0;
}

int LaneDet::detect_and_draw(const cv::Mat& rgb) {
  if (!loaded_) return -1;

  // data_transform = transforms.Compose([
  //     transforms.Resize((256,  512)),
  //     transforms.ToTensor(),
  //     # output[channel] = (input[channel] - mean[channel]) / std[channel]
  //     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  // ])
  int resized_width = 512;
  int resized_height = 256;
  ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, rgb.cols,
                                                  rgb.rows, resized_width, resized_height);

  // normalize the image
  float mean[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
  float norm[3] = {0.01712475383f, 0.0175070028, 0.01742919389};
  input.substract_mean_normalize(mean, norm);

  // inference the image
  ncnn::Extractor ex = lanedet.create_extractor();

  ex.input("in0", input);

  ncnn::Mat feat;
  ex.extract("out0", feat);

  // draw the feat to image
  // feat shape: 3 1 256 512
  // LOGI("feat shape: %d %d %d %d", feat.c, feat.d, feat.h, feat.w);

  const float* ptr_ret_0 = feat.channel(0);
  const float* ptr_ret_1 = feat.channel(1);
  const float* ptr_r = feat.channel(2);
  const float* ptr_g = feat.channel(3);
  const float* ptr_b = feat.channel(4);

  float factor_w = static_cast<float>(rgb.cols) / resized_width;
  float factor_h = static_cast<float>(rgb.rows) / resized_height;

  for (int y = 0; y < feat.h; y++) {
    for (int x = 0; x < feat.w; x++) {
      if (ptr_ret_0[x] > ptr_ret_1[x]) {
        continue;
      }
      cv::circle(rgb, cv::Point(factor_w * x, factor_h * y), 2,
                 cv::Scalar(255 * ptr_r[x], 255 * ptr_g[x], 255 * ptr_b[x]), -1);
    }
    ptr_ret_0 += feat.w;
    ptr_ret_1 += feat.w;
    ptr_r += feat.w;
    ptr_g += feat.w;
    ptr_b += feat.w;
  }

  return 1;
}
