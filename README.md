# ncnn-android-nanodet

The NanoDet object detection

This is a sample ncnn android project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile

## android apk file download
https://github.com/nihui/ncnn-android-nanodet/releases/latest

## how to build and run

install java 17
```
apt install openjdk-17-jdk openjdk-17-jre
```

### step1
https://github.com/Tencent/ncnn/releases

* Download [ncnn-YYYYMMDD-android-vulkan.zip](https://github.com/Tencent/ncnn/releases/download/20240820/ncnn-20240820-android-vulkan-shared.zip) or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download [opencv-mobile-XYZ-android.zip](https://github.com/nihui/opencv-mobile/releases/latest/download/opencv-mobile-4.10.0-android.zip)
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
* Open this project with Android Studio, build it and enjoy!

## some notes
* Android ndk camera is used for best efficiency
* Crash may happen on very old devices for lacking HAL3 camera interface
* All models are manually modified to accept dynamic input shape
* Most small models run slower on GPU than on CPU, this is common
* FPS may be lower in dark environment because of longer camera exposure time

## screenshot
![](screenshot.jpg)
