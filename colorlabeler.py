import colorsys
import extcolors
from PIL import Image, ImageDraw
import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2lab, deltaE_cie76
import random
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
                    color=(255, 255, 255), thickness=1)
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


def analize_image_zones(img, colors):

    for c in colors:
        hsv_color_upper = (c[1][0] + 10, c[1][1] + 40, c[1][2] + 40)
        hsv_color_lower = (c[1][0] - 10, c[1][1] - 40, c[1][2] - 40)
        mask = cv2.inRange(img, hsv_color_lower, hsv_color_upper)
        croped = cv2.bitwise_and(img, img, mask=mask)
        cv2.putText(croped, f"{c} color of {len(colors)}", (10, 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(255, 255, 255), thickness=1)
        cv2.imshow("croped", croped)
        cv2.waitKey()


def main():

    global image_colors
    img_cv = cv2.imread("mario.png")

    show_colors_list(img_cv)

    image_colors = extract_image_colors(img_cv)
    opened_img = opening_colors(img_cv, image_colors, 3)
    cv2.imshow("opened_image", opened_img)
    cv2.waitKey(0)

    image_colors = extract_image_colors_rgb(opened_img)
    unified_img = unify_similar_colors(opened_img, image_colors, 40)
    cv2.imshow("unified_image", unified_img)
    cv2.waitKey(0)

    show_colors_list(unified_img)

    image_colors = extract_image_colors(unified_img)
    analize_image_zones(unified_img, image_colors)

    #cv2.namedWindow('unified_image')
    #cv2.setMouseCallback('unified_image', pick_color)

    # now click into the hsv img , and look at values:
    #image_colors = cv2.cvtColor(unified_img, cv2.COLOR_BGR2RGB)
    #cv2.imshow("unified_image", image_colors)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

