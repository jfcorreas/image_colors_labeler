import cv2 as cv

erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_type = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_erode_kernel_size = 'Erode Kernel size:\n 2n +1'
title_trackbar_dilate_kernel_size = 'Dilate Kernel size:\n 2n +1'
title_dilate_erode_window = 'Dilatation-Erosion Demo'
result_img = None
src_img = None


def dilatation_erosion(val):
    global result_img
    global src_img

    dilatation_size = cv.getTrackbarPos(title_trackbar_dilate_kernel_size, title_dilate_erode_window)
    erosion_size = cv.getTrackbarPos(title_trackbar_erode_kernel_size, title_dilate_erode_window)
    morph_type = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_dilate_erode_window)
    if val_type == 0:
        morph_type = cv.MORPH_RECT
    elif val_type == 1:
        morph_type = cv.MORPH_CROSS
    elif val_type == 2:
        morph_type = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(morph_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src_img, element)
    element = cv.getStructuringElement(morph_type, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    result_img = cv.erode(dilatation_dst, element)
    cv.imshow(title_dilate_erode_window, result_img)


def dilate_and_erode(src):
    global result_img
    global src_img

    result_img = src.copy()
    src_img = src.copy()

    cv.namedWindow(title_dilate_erode_window)
    cv.createTrackbar(title_trackbar_element_type, title_dilate_erode_window, 0, max_elem, dilatation_erosion)
    cv.createTrackbar(title_trackbar_dilate_kernel_size, title_dilate_erode_window, 0, max_kernel_size, dilatation_erosion)
    cv.createTrackbar(title_trackbar_erode_kernel_size, title_dilate_erode_window, 0, max_kernel_size, dilatation_erosion)
    dilatation_erosion(0)
    cv.waitKey()
    return result_img
