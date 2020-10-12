import cv2
import numpy as np

img = cv2.imread("mario.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey()
retval, threshed = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)


contours, h = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

dst1 = 255 * np.ones(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
dst2 = img.copy()
cv2.drawContours(dst1, contours, -1, (0, 255, 0), 1)

cnts = []
for cnt in contours:
    rect = cv2.boundingRect(cnt)
    x, y, w, h = rect
    if w < 10 or h < 10 >> w > 100 or h > 100:
        continue
    cnts.append(cnt)
    pixel = img[y + int(h/2), x + int(w/2)]
    col = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
    cv2.putText(dst1, "Z", (x + int(w/2), y + int(h/2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2,
                color=tuple(col), thickness=1)

print(len(cnts))
cv2.drawContours(dst2, cnts, -1, (0,255,0), 3)

res = np.hstack((dst1, dst2))
cv2.imshow("res", res)
cv2.waitKey()
cv2.destroyAllWindows()