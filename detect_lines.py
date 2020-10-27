import cv2 as cv
import numpy as np

max_line_length = 50
max_line_gap_limit = 50
window_name = 'Pattern Map'
title_trackbar_length = 'Min Line Length:'
title_trackbar_gap = 'Max Line Gap:'
result_img = None
src_img = None


def is_vertical_line(x0, y0, x1, y1):
    t5 = np.tan(5 * np.pi / 180)

    dx = x1-x0
    dy = y1-y0

    if dy != 0 and abs(dx / dy) < t5:
        return True
    else:
        return False


def hough_lines(val):
    global result_img
    global src_img

    min_line_length = cv.getTrackbarPos(title_trackbar_length, window_name)
    max_line_gap = cv.getTrackbarPos(title_trackbar_gap, window_name)
    lines = cv.HoughLinesP(src_img, 1, np.pi / 180, 50, min_line_length, max_line_gap)

    if len(lines) == 0:
        print('No lines were found')
        exit()

    filtered_lines = lines
    height, width = src_img.shape
    result_img = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

    if filtered_lines is not None:
        for i in range(0, len(filtered_lines)):
            l = filtered_lines[i][0]
            if is_vertical_line(l[0], l[1], l[2], l[3]):
                cv.line(result_img, (l[0], 0), (l[2], height-1), (0, 0, 0), 2)
            else:
                cv.line(result_img, (0, l[1]), (width-1, l[3]), (0, 0, 0), 2)
    cv.imshow(window_name, result_img)


def detect_lines(src):
    global result_img
    global src_img

    result_img = src.copy()
    src_img = src.copy()
    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar_length, window_name, 0, max_line_length, hough_lines)
    cv.createTrackbar(title_trackbar_gap, window_name, 0, max_line_gap_limit, hough_lines)
    hough_lines(0)
    cv.waitKey()
    return result_img
