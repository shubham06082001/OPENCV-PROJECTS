import numpy as np
import cv2

img = cv2.imread("shapes.jpg")
img_contour = img.copy()


def get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        # cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
        if area > 500:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            print(perimeter)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            print(len(approx))
            obj_cor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if obj_cor == 3:
                obj_type = "tri"
            elif obj_cor == 4:
                asp_ratio = w / float(h)
                if 0.95 < asp_ratio < 1.05:
                    obj_type = "square"
                else:
                    obj_type = "rectangle"
            elif obj_cor == 8:
                obj_type = "octagon"
            elif obj_cor > 4:
                obj_type = "circle"
            else:
                obj_type = "none"

            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_contour, obj_type,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
img_canny = cv2.Canny(img_blur, 50, 50)
img_blank = np.zeros_like(img)
get_contours(img_canny)

img_hor = np.hstack((img_gray, img_blur))

# cv2.imshow("original", img)
# cv2.imshow("stack", img_hor)
cv2.imshow("canny", img_canny)
cv2.imshow("contour", img_contour)
cv2.waitKey(0)
