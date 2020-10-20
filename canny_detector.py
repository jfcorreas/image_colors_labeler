import cv2 as cv

max_lowThreshold = 300
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
canny_img = None
src_img = None


def canny_threshold(val):
    global canny_img
    global src_img

    low_threshold = val
    img_blur = cv.blur(src_img, (3, 3))
    canny_img = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    cv.imshow(window_name, canny_img)


def canny_image(src):
    global canny_img
    global src_img

    canny_img = src.copy()
    src_img = src.copy()
    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, canny_threshold)
    canny_threshold(0)
    cv.waitKey()
    return canny_img



