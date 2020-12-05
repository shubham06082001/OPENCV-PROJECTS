import cv2
import numpy as np


def empty():
    pass


cv2.namedWindow("trackbars")
cv2.resizeWindow("trackbars", 640, 240)
cv2.createTrackbar("hue min", "trackbars", 0, 179, empty)
cv2.createTrackbar("hue max", "trackbars", 19, 179, empty)
cv2.createTrackbar("sat min", "trackbars", 110, 255, empty)
cv2.createTrackbar("sat max", "trackbars", 240, 255, empty)
cv2.createTrackbar("val min", "trackbars", 153, 255, empty)
cv2.createTrackbar("val max", "trackbars", 255, 255, empty)

while True:
    img = cv2.imread("girl.jpg")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("hue min", "trackbars")
    h_max = cv2.getTrackbarPos("hue max", "trackbars")
    s_min = cv2.getTrackbarPos("sat min", "trackbars")
    s_max = cv2.getTrackbarPos("sat max", "trackbars")
    v_min = cv2.getTrackbarPos("val min", "trackbars")
    v_max = cv2.getTrackbarPos("val max", "trackbars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_result = cv2.bitwise_and(img, img, mask=mask)

    # img_hor1 = np.hstack((img, mask))
    img_hor2 = np.hstack((img_hsv, img_result))

    # cv2.putText(img_hsv, "hsv image", (300, 200), cv2.FONT_ITALIC, 1, (255, 150, 70), 1)
    # cv2.putText(img_result, "result image", (300, 200), cv2.FONT_ITALIC, 1, (140, 250, 170), 1)
    cv2.putText(mask, "masked image", (100, 200), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)

    # cv2.imshow("original", img)
    # cv2.imshow("hsv", img_hsv)
    cv2.imshow("mask", mask)
    # cv2.imshow("result", img_result)
    # cv2.imshow("stack 1", img_hor1)
    cv2.imshow("stack 2", img_hor2)

    cv2.waitKey(1)
