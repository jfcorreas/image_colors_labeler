import cv2 as cv

max_symbol_offset = 50
max_font_size = 20
window_name = 'Colors Map'
title_symbol_offset_y = 'Y Position:'
title_symbol_offset_x = 'X Position:'
title_font_size = 'Font Size:'
src_grid_img = None
result_grid = None
src_img = None
img_contours = None
color_symbols = None


def generate_color_symbol_dict(image_colors):
    symbols = list("AX<OT#5P·V/B9$@2L=N+")
    color_symbol_dict = {}
    for color in image_colors:
        c = (color[0], color[1], color[2])
        color_symbol_dict[tuple(c)] = symbols.pop()
    return color_symbol_dict


def find_grid_contours(grid):
    retval, threshed = cv.threshold(grid, 0, 255, cv.THRESH_BINARY)

    contours, h = cv.findContours(threshed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    return contours


def mark_image_colors(val):
    global src_grid_img
    global src_img
    global result_grid
    global img_contours
    global color_symbols

    offset_x = cv.getTrackbarPos(title_symbol_offset_x, window_name)
    offset_y = cv.getTrackbarPos(title_symbol_offset_y, window_name)
    font_size = cv.getTrackbarPos(title_font_size, window_name)

    result_grid = src_grid_img.copy()

    for cnt in img_contours:
        rect = cv.boundingRect(cnt)
        x, y, w, h = rect
        pixel = src_img[y + int(h / 2), x + int(w / 2)]
        col = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        cv.putText(result_grid, color_symbols[tuple(col)], (x + offset_x, y + offset_y),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size/10,
                    color=tuple(col), thickness=1)
    cv.imshow(window_name, result_grid)


def tag_image_colors(src, src_grid, src_colors):
    global src_grid_img
    global src_img
    global result_grid
    global img_contours
    global color_symbols

    src_grid_img = src_grid.copy()
    src_img = src.copy()
    img_contours = find_grid_contours(cv.cvtColor(src_grid, cv.COLOR_BGR2GRAY))
    color_symbols = generate_color_symbol_dict(src_colors)

    cv.namedWindow(window_name)
    cv.createTrackbar(title_symbol_offset_x, window_name, 1, max_symbol_offset, mark_image_colors)
    cv.createTrackbar(title_symbol_offset_y, window_name, 1, max_symbol_offset, mark_image_colors)
    cv.createTrackbar(title_font_size, window_name, 1, max_font_size, mark_image_colors)
    mark_image_colors(0)
    cv.waitKey()
    return result_grid
