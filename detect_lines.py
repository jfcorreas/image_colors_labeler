import cv2 as cv
import numpy as np

max_line_gap_limit = 30
window_name = 'Pattern Map'
title_trackbar_gap = 'Max Line Gap:'
result_img = None
src_img = None
vertical_lines = None
horizontal_lines = None


def is_vertical_line(x0, y0, x1, y1):
    t5 = np.tan(5 * np.pi / 180)

    dx = x1-x0
    dy = y1-y0

    if dy != 0 and abs(dx / dy) < t5:
        return True
    else:
        return False


def hough_lines(val):
    global src_img
    global vertical_lines
    global horizontal_lines

    min_line_length = 10
    max_line_gap = cv.getTrackbarPos(title_trackbar_gap, window_name)

    vertical_lines = set()
    horizontal_lines = set()

    lines = cv.HoughLinesP(src_img, 1, np.pi / 180, 50, min_line_length, max_line_gap)

    if len(lines) == 0:
        print('No lines were found')
        exit()

    filtered_lines = lines
    height, width = src_img.shape
    hough_img = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

    if filtered_lines is not None:
        for i in range(0, len(filtered_lines)):
            l = filtered_lines[i][0]
            if is_vertical_line(l[0], l[1], l[2], l[3]):
                cv.line(hough_img, (l[0], 0), (l[2], height-1), (0, 0, 0), 2)
                vertical_lines.add(l[0])
            else:
                cv.line(hough_img, (0, l[1]), (width-1, l[3]), (0, 0, 0), 2)
                horizontal_lines.add(l[1])
    cv.imshow(window_name, hough_img)


def get_draw_guide(coord: list):
    guide = [None]*2
    min_difference = coord[len(coord)-1]*10
    last_point = 0
    for point in coord:
        diff = point - last_point
        if min_difference is not None and diff < min_difference:
            guide[0] = point
            guide[1] = diff
            min_difference = diff
        last_point = point
    return guide


def print_lines(vertical: bool, lower_limit: int, upper_limit: int,
                line_length: int, guide: list):
    global result_img

    point = guide[0]
    diff = guide[1]

    while point >= lower_limit + diff:
        point -= diff

    while point < upper_limit:
        if vertical:
            cv.line(result_img, (point, 0), (point, line_length), (0, 0, 0), 1)
        else:
            cv.line(result_img, (0, point), (line_length, point), (0, 0, 0), 1)
        point += diff


def make_grid(guidex, guidey):
    global result_img
    global src_img

    height, width = src_img.shape
    result_img = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

    print_lines(True, 0, width, height, guidex)
    print_lines(False, 0, height, width, guidey)


def detect_lines(src):
    global result_img
    global src_img
    global vertical_lines
    global horizontal_lines

    src_img = src.copy()
    cv.namedWindow(window_name)
    cv.createTrackbar(title_trackbar_gap, window_name, 10, max_line_gap_limit, hough_lines)
    hough_lines(0)
    cv.waitKey()

    guidex = get_draw_guide(sorted(vertical_lines))
    guidey = get_draw_guide((sorted(horizontal_lines)))

    make_grid(guidex, guidey)

    return result_img
