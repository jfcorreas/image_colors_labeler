import colorsys
import extcolors
from PIL import Image, ImageDraw
import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2lab, deltaE_cie76
import random
import string
import matplotlib
import matplotlib.pyplot as plt


image_colors = None  # global


def extract_image_colors(img):
    colors = Image.fromarray(img).getcolors()
    return colors


def extract_image_colors_rgb(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = Image.fromarray(rgb_image).getcolors()
    return colors


def unify_similar_colors(img, colors, delta_threshold: int):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_image_array = np.array(rgb_image)

    lab = rgb2lab(rgb_image)

    for color in colors:
        color_3d = np.uint8(np.asarray([[color[1]]]))
        dE_color = deltaE_cie76(rgb2lab(color_3d), lab)
        rgb_image_array[dE_color < delta_threshold] = color_3d

    rgb_image = Image.fromarray(rgb_image_array, 'RGB')
    rgb_image = np.uint8(rgb_image)

    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)


def opening_colors(img, colors, radius: int):

    combined_mask = 0

    for c in colors:
        mask = cv2.inRange(img, c[1], c[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        combined_mask = combined_mask | opened_mask

    masked_image = cv2.bitwise_and(img, img, mask=combined_mask)

    return masked_image


def show_colors_list(img):
    matplotlib.use('TkAgg')
    whiteblankimage = 255 * np.ones(shape=[500, 500, 3], dtype=np.uint8)
    x = 0
    y = 500
    for c in extract_image_colors_rgb(img):
        rgb_color = (c[1][0], c[1][1], c[1][2])
        cv2.rectangle(whiteblankimage, pt1=(x, y), pt2=(x+100, y-100), color=rgb_color, thickness=-1)
        cv2.putText(whiteblankimage, f"{rgb_color}", (x+5, y-40),
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


def fill_image_with_text(img, text: str, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    y_offset = int(h / 30)
    x_offset = int(w / 30)

    for y in range(0, h, y_offset):
        for x in range(0, w, x_offset):
            cv2.putText(img, text, (x, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=color, thickness=1)

    return img


def mark_image_colors(img, colors):

    chars = list(string.ascii_uppercase[0:len(colors)])
    analized_img = 0

    for c in colors:
        hsv_color_upper = (c[1][0] + 10, c[1][1] + 40, c[1][2] + 40)
        hsv_color_lower = (c[1][0] - 10, c[1][1] - 40, c[1][2] - 40)
        mask = cv2.inRange(img, hsv_color_lower, hsv_color_upper)
        mask_filled = fill_image_with_text(mask, chars.pop())
        croped = cv2.bitwise_and(img, img, mask=mask_filled)
        #cv2.putText(croped, f"{c} color of {len(colors)}", (10, 10),
        #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
        #            color=(255, 255, 255), thickness=1)
        analized_img = analized_img | croped

    return analized_img


def find_grid(img):

    filter = False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    cv2.imshow('canny.jpg', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if not lines.any():
        print('No lines were found')
        exit()

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

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
            if not line_flags[indices[i]]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('hough.jpg', img)


def main():

    global image_colors
    img_cv = cv2.imread("mario2.jpg", cv2.IMREAD_COLOR)
    find_grid(img_cv)
    cv2.waitKey(0)

    show_colors_list(img_cv)

    image_colors = extract_image_colors(img_cv)
    opened_img = opening_colors(img_cv, image_colors, 3)
    cv2.imshow("opened_image", opened_img)
    cv2.waitKey(0)

    image_colors = extract_image_colors_rgb(opened_img)
    unified_img = unify_similar_colors(opened_img, image_colors, 4)
    cv2.imshow("unified_image", unified_img)
    cv2.waitKey(0)

    show_colors_list(unified_img)

    image_colors = extract_image_colors(unified_img)
    analyzed_image = mark_image_colors(unified_img, image_colors)

    #cv2.namedWindow('unified_image')
    #cv2.setMouseCallback('unified_image', pick_color)

    # now click into the hsv img , and look at values:
    #image_colors = cv2.cvtColor(unified_img, cv2.COLOR_BGR2RGB)
    #cv2.imshow("unified_image", image_colors)

    cv2.imshow("Analyzed Image", analyzed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

