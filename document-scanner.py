import cv2
import numpy as np

width = 480
height = 640
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(10, 150)


def preProcessing(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5, 5))
    img_dialation = cv2.dilate(img_canny, kernel, iterations=2)
    img_threshold = cv2.erode(img_dialation, kernel, iterations=1)

    return img_threshold


def get_contours(image):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), np.int32)
    add = my_points.sum(1)
    # print("add", add)

    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]
    # print("new points", my_points_new)

    return my_points_new


def get_warp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width, height))

    img_cropped = img_output[20:img_output.shape[0] - 20, 20:img_output.shape[1] - 20]
    img_cropped = cv2.resize(img_cropped, (width, height))

    return img_cropped


while True:
    success, img = cap.read()
    cv2.resize(img, (width, height))
    img_contour = img.copy()

    img_thres = preProcessing(img)
    biggest = get_contours(img_thres)
    # print(biggest)
    if biggest.size != 0:
        img_warped = get_warp(img, biggest)
        cv2.imshow("warped image", img_warped)
        cv2.imshow("contour image", img_contour)
        cv2.imshow("threshold image", img_thres)
    else:
        cv2.imshow("original image", img)
        cv2.imshow("threshold image", img_thres)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
