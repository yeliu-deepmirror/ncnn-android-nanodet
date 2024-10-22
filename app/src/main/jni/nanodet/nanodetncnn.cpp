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

#include <android/asset_manager_jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <benchmark.h>
#include <platform.h>

#include "nanodet.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON

static NanoDet* g_nanodet = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow {
 public:
  virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const {
  // LOGI("image_size %d x %d", rgb.cols, rgb.rows);
  // nanodet
  {
    ncnn::MutexLockGuard g(lock);

    if (g_nanodet) {
      std::vector<Object> objects;
      g_nanodet->detect(rgb, objects);

      g_nanodet->draw(rgb, objects);
    } else {
      draw_unsupported(rgb);
    }
  }

  draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  LOGI("JNI_OnLoad");

  g_camera = new MyNdkCamera;

  return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
  LOGI("JNI_OnUnload");

  {
    ncnn::MutexLockGuard g(lock);

    delete g_nanodet;
    g_nanodet = 0;
  }

  delete g_camera;
  g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetncnn_NanoDetNcnn_loadModel(
    JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu) {
  if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
    return JNI_FALSE;
  }

  AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
  LOGI("loadModel %p", mgr);

  const char* modeltypes[] = {"m",          "m-416",        "g", "ELite0_320", "ELite1_416",
                              "ELite2_512", "RepVGG-A0_416"

  };

  const int target_sizes[] = {320, 416, 416, 320, 416, 512, 416};

  const float mean_vals[][3] = {{103.53f, 116.28f, 123.675f}, {103.53f, 116.28f, 123.675f},
                                {103.53f, 116.28f, 123.675f}, {127.f, 127.f, 127.f},
                                {127.f, 127.f, 127.f},        {127.f, 127.f, 127.f},
                                {103.53f, 116.28f, 123.675f}};

  const float norm_vals[][3] = {
      {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f}, {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f},
      {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f}, {1.f / 128.f, 1.f / 128.f, 1.f / 128.f},
      {1.f / 128.f, 1.f / 128.f, 1.f / 128.f},      {1.f / 128.f, 1.f / 128.f, 1.f / 128.f},
      {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f}};

  const char* modeltype = modeltypes[(int)modelid];
  int target_size = target_sizes[(int)modelid];
  bool use_gpu = (int)cpugpu == 1;

  // reload
  {
    ncnn::MutexLockGuard g(lock);

    if (use_gpu && ncnn::get_gpu_count() == 0) {
      // no gpu
      delete g_nanodet;
      g_nanodet = 0;
    } else {
      if (!g_nanodet) g_nanodet = new NanoDet;
      g_nanodet->load(mgr, modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid],
                      use_gpu);
    }
  }

  return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetncnn_NanoDetNcnn_openCamera(JNIEnv* env,
                                                                               jobject thiz,
                                                                               jint facing) {
  if (facing < 0 || facing > 1) return JNI_FALSE;
  LOGI("openCamera %d", facing);

  g_camera->open((int)facing);

  return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetncnn_NanoDetNcnn_closeCamera(JNIEnv* env,
                                                                                jobject thiz) {
  LOGI("closeCamera");

  g_camera->close();

  return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetncnn_NanoDetNcnn_setOutputWindow(
    JNIEnv* env, jobject thiz, jobject surface) {
  ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
  LOGI("setOutputWindow %p", win);

  g_camera->set_window(win);

  return JNI_TRUE;
}
}
