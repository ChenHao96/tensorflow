#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2

from raspberrypi.image import record_image_array


def record_video_capture(capture, files, index):
    success, frame = capture.read()
    if success:
        return record_video_frame(frame, files, index)
    return 400, 300


def record_video_frame(frame, files, index):
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    return record_image_array(cv2image, files, index)


def record_video_capture_2_array(capture, rgba=True):
    success, frame = capture.read()
    if success:
        frame = cv2.flip(frame, 1)
        if rgba:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            return True, cv2image
        return True, frame
    return False, None
