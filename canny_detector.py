from __future__ import print_function
import cv2 as cv

max_lowThreshold = 300
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3


def canny_image(src):
    def canny_threshold(val):
        low_threshold = val
        img_blur = cv.blur(src, (3, 3))
        detected_edges = cv.Canny(src, low_threshold, low_threshold * ratio, kernel_size)
        #mask = detected_edges != 0
        #dst = src * (mask[:, :, None].astype(src.dtype))
        cv.imshow(window_name, detected_edges)
        return detected_edges

    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, canny_threshold)
    canny_img = canny_threshold(0)
    cv.waitKey()
    return canny_img

