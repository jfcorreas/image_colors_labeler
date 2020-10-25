import math

from PIL import Image, ImageDraw
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76
import matplotlib
import matplotlib.pyplot as plt

from canny_detector import canny_image
from dilation_erosion import dilate_and_erode

image_colors = None


def extract_image_colors(img):
    all_rgb_codes = img.reshape(-1, img.shape[-1]).copy()
    colors = np.unique(all_rgb_codes, axis=0, return_counts=False)
    return colors


def replace_black_color(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    img[mask == 255] = [5, 5, 5]
    return img


def opening_colors(img, radius: int):
    combined_mask = 0

    for c in image_colors:
        mask = cv2.inRange(img, c, c)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        combined_mask = combined_mask | opened_mask

    masked_image = cv2.bitwise_and(img, img, mask=combined_mask)

    return masked_image


def unify_similar_colors(img, delta_threshold: int):
    rgb_image_array = np.array(img)

    lab = rgb2lab(img)

    for color in image_colors:
        color_3d = np.uint8(np.asarray([[color]]))
        dE_color = deltaE_cie76(rgb2lab(color_3d), lab)
        rgb_image_array[dE_color < delta_threshold] = color_3d

    rgb_image = Image.fromarray(rgb_image_array, 'RGB')
    rgb_image = np.uint8(rgb_image)

    return rgb_image


def show_colors_list(img):
    matplotlib.use('TkAgg')
    whiteblankimage = 255 * np.ones(shape=[500, 500, 3], dtype=np.uint8)
    x = 0
    y = 500
    for c in extract_image_colors(img):
        rgb_color = (int(c[2]), int(c[1]), int(c[0]))  # B, G, R conversion
        cv2.rectangle(whiteblankimage, pt1=(x, y), pt2=(x + 100, y - 100), color=rgb_color, thickness=-1)
        cv2.putText(whiteblankimage, f"{rgb_color}", (x + 5, y - 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(200, 250, 200), thickness=1)
        if x < 400:
            x += 100
        else:
            x = 0
            y -= 100

    plt.imshow(whiteblankimage)

    plt.show()


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_colors[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 1, pixel[1] + 1, pixel[2] + 1])
        lower = np.array([pixel[0] - 1, pixel[1] - 1, pixel[2] - 1])
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_colors, lower, upper)
        cv2.imshow("mask", image_mask)


def filter_lines2(lines):
    for x1, y1, x2, y2 in lines:
        for index, (x3, y3, x4, y4) in enumerate(lines):

            if y1 == y2 and y3 == y4:  # Horizontal Lines
                diff = abs(y1 - y3)
            elif x1 == x2 and x3 == x4:  # Vertical Lines
                diff = abs(x1 - x3)
            else:
                diff = 0

            if diff < 10 and diff != 0:
                del lines[index]
    return lines


def filter_lines(lines, rho_threshold=40, theta_threshold=0.5):
    # how many lines are similar to a given one
    similar_lines = {i: [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x: len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines) * [True]
    for i in range(len(lines) - 1):
        if not line_flags[indices[i]]:  # if we already disregarded the ith element in the ordered list then we don't
            # care (we will not delete anything based on it and we will never reconsider using this line again)
            continue

        for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
            if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                continue

            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[
                    indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

    filtered_lines = []

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])
    return filtered_lines


def find_grid(img, low_threshold=0, ratio=33, kernel_size=3):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = canny_image(gray)
    edges = dilate_and_erode(edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=10)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 150).tolist()

    if len(lines) == 0:
        print('No lines were found')
        exit()

    print('number of Hough lines:', len(lines))

    filtered_lines = lines
    #filtered_lines = filter_lines2(lines)

    grid_pattern = 255 * np.ones(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)

    print('Number of filtered lines:', len(filtered_lines))
    # Draw the lines
    if filtered_lines is not None:
        for i in range(0, len(filtered_lines)):
            l = filtered_lines[i][0]
            cv2.line(grid_pattern, (l[0], l[1]), (l[2], l[3]), (0, 0, 0), 3, 2)

    #if filtered_lines is not None:
    #    for i in range(0, len(filtered_lines)):
    #        rho = lines[i][0][0]
    #        theta = lines[i][0][1]
    #        a = np.cos(theta)
    #        b = np.sin(theta)
    #        x0 = a * rho
    #        y0 = b * rho
    #        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #        cv2.line(grid_pattern, pt1, pt2, (0, 0, 0), 3, 2)

    return grid_pattern


def find_grid_contours(grid):
    gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    retval, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    contours, h = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def mark_image_colors(grid, img, color_symbols):
    contours = find_grid_contours(grid)
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        pixel = img[y + int(h / 2), x + int(w / 2)]
        col = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        cv2.putText(grid, color_symbols[tuple(col)], (x + 5, y + 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=tuple(col), thickness=1)
    return grid


def generate_color_symbol_dict(image_colors):
    symbols = list("AXOT#5PVB9$º=@2LN/+")
    color_symbol_dict = {}
    for color in image_colors:
        c = (color[0], color[1], color[2])
        color_symbol_dict[tuple(c)] = symbols.pop()
    return color_symbol_dict


def main():
    global image_colors

    img_path = "mario.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img_without_black = replace_black_color(img)
    cv2.imshow("image without black", img_without_black)
    # cv2.waitKey(0)

    image_colors = extract_image_colors(img_without_black)
    opened_img = opening_colors(img_without_black, 3)
    # cv2.imshow("opened_image", opened_img)

    image_colors = extract_image_colors(opened_img)
    unified_img = unify_similar_colors(opened_img, 4)
    # cv2.imshow("unified_image", unified_img)
    # cv2.waitKey(0)

    grid_pattern = find_grid(unified_img)
    cv2.imshow("Grid Pattern", grid_pattern)

    image_colors = extract_image_colors(unified_img)
    color_symbols = generate_color_symbol_dict(image_colors)
    analyzed_grid = mark_image_colors(grid_pattern, unified_img, color_symbols)
    cv2.imshow("Analyzed Grid Pattern", analyzed_grid)


    #show_colors_list(unified_img)

    # cv2.namedWindow('unified_image')
    # cv2.setMouseCallback('unified_image', pick_color)

    # now click into the hsv img , and look at values:
    # image_colors = cv2.cvtColor(unified_img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("unified_image", image_colors)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
