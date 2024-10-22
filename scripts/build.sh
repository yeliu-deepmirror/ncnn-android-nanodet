#!/usr/bin/env bash
set -e

./gradlew assembleDebug
adb install /ncnn-android-nanodet/app/build/outputs/apk/debug/com.tencent.nanodetncnn-debug.apk
