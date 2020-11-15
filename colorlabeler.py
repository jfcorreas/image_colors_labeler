from PIL import Image
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76
import matplotlib
import matplotlib.pyplot as plt

from canny_detector import canny_image
from detect_lines import detect_lines
from dilation_erosion import dilate_and_erode
from tag_colors import tag_image_colors

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


def unify_similar_colors(img, delta_threshold: int):
    rgb_image_array = np.array(img)

    lab = rgb2lab(img)

    for color in image_colors:
        print(".")
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


def find_grid(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = canny_image(gray)
    edges = dilate_and_erode(edges)
    grid_pattern = detect_lines(edges)

    grid_pattern = dilate_and_erode(grid_pattern)

    return grid_pattern


def main():
    global image_colors

    img_path = "images/mario-kart_16c.png"
    grid_path = "images/mario-kart_grid.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    grid_img = cv2.imread(grid_path, cv2.IMREAD_COLOR)

    img_without_black = replace_black_color(img)
    cv2.imshow("without_black", img_without_black)
    cv2.waitKey(20)
    image_colors = extract_image_colors(img_without_black)

    unified_img = unify_similar_colors(img_without_black, 4)
    image_colors = extract_image_colors(unified_img)
    cv2.imshow(f"unified_image: {len(image_colors)} colors", unified_img)
    cv2.waitKey(20)

    grid_pattern = find_grid(grid_img)

    analyzed_grid = tag_image_colors(unified_img, grid_pattern, image_colors)
    cv2.imshow("Analyzed Grid Pattern", analyzed_grid)
    cv2.waitKey(20)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
