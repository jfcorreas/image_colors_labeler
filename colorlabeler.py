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


def extract_image_colors(img):
    colors = img.convert('RGB').getcolors()
    return colors


def extract_image_colors2(img):
    colors, pixel_count = extcolors.extract_from_image(img)
    return colors


def unify_similar_colors(img, delta_threshold: int):
    rgb_image = img.convert('RGB')
    rgb_image_array = np.array(rgb_image)

    lab = rgb2lab(rgb_image)

    for color in extract_image_colors(img):
        color_3d = np.uint8(np.asarray([[color[1]]]))
        print(color_3d)
        dE_color = deltaE_cie76(rgb2lab(color_3d), lab)
        rgb_image_array[dE_color < delta_threshold] = color_3d

    rgb_image = Image.fromarray(rgb_image_array, 'RGB')
    return rgb_image


def show_colors_list(img):
    matplotlib.use('TkAgg')
    whiteblankimage = 255 * np.ones(shape=[500, 500, 3], dtype=np.uint8)
    x = 0
    y = 500
    for c in extract_image_colors(img):
        rgb_color = (c[1][0], c[1][1], c[1][2])
        print(f"RGB color: {rgb_color}")
        cv2.rectangle(whiteblankimage, pt1=(x, y), pt2=(x+100, y-100), color=rgb_color, thickness=-1)
        cv2.putText(whiteblankimage, f"{rgb_color}", (x+5, y-40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(255, 255, 255), thickness=2)
        if x < 400:
            x += 100
        else:
            x = 0
            y -= 100

    plt.imshow(whiteblankimage)

    plt.show()


def analize_image_zones():
    img_cv = cv2.imread("mario.png")
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    #d = ImageDraw.Draw(img)
    #d.text((10, 10), f"{len(colors)} detected in image")

    #img.save('pil_text.png')
    for c in colors:
        hsv_color_upper = (c[1][0] + 10, c[1][1] + 40, c[1][2] + 40)
        hsv_color_lower = (c[1][0] - 10, c[1][1] - 40, c[1][2] - 40)
        mask = cv2.inRange(img_cv, hsv_color_lower, hsv_color_upper)
        croped = cv2.bitwise_and(img_cv, img_cv, mask=mask)
        cv2.imshow("croped", croped)
        cv2.waitKey()


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def main():

    img = Image.open("mario.png")

    unified_img = unify_similar_colors(img, 40)

   # cv2.imshow("unified", unified_img)
   # cv2.waitKey()

    show_colors_list(unified_img)


if __name__ == '__main__':
    main()

