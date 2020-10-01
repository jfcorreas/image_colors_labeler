from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def analize_image_colors():
    img = Image.open("mario.png")
    # colors, pixel_count = extcolors.extract_from_image(img)
    colors = img.convert('HSV').getcolors()
    print(colors)
    #print(pixel_count)

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


def main():
    matplotlib.use('TkAgg')
    whiteblankimage = 255 * np.ones(shape=[500, 500, 3], dtype=np.uint8)
    x = 0
    y = 500
    while x < 500:
        while y > 0:
            cv2.rectangle(whiteblankimage, pt1=(x, y), pt2=(x+100, y-100), color=(0, 0, 255), thickness=-1)
            y -= 100
        x += 100
        y = 500

    plt.imshow(whiteblankimage)

    plt.show()


if __name__ == '__main__':
    main()

