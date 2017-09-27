import numpy as np
import cv2


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
def lab_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def median_blur(image, ksize=3):
    return cv2.medianBlur(image, ksize)


cap = cv2.VideoCapture('input/video.avi')

scale = 1.7

first_frame = None

while cap.isOpened():
    _, frame = cap.read()

    if frame is None:
        break

    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    gamma_frame = median_blur(adjust_gamma(frame, gamma=1))
    contrast_frame = median_blur(lab_contrast(gamma_frame), 5)
    gray_frame = median_blur(cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2GRAY))

    if first_frame is None:
        first_frame = gray_frame
        continue

    delta = cv2.absdiff(first_frame, gray_frame)

    _, thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=3)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_frame = frame.copy()

    max_cnt = None
    for cnt in contours:
        if max_cnt is None:
            max_cnt = cnt
            continue
        if cv2.contourArea(cnt) > cv2.contourArea(max_cnt):
            max_cnt = cnt

    hull = cv2.convexHull(max_cnt, returnPoints=False)
    cv2.drawContours(result_frame, [hull], -1, (255, 255, 255), 2)

    defects = cv2.convexityDefects(max_cnt, hull)

    epsilon = 0.01 * cv2.arcLength(max_cnt, True)
    approx = cv2.approxPolyDP(max_cnt, epsilon, True)
    cv2.drawContours(result_frame, [approx], -1, (255, 255, 255), 1)

    if defects is not None:
        far_defect = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_cnt[s][0])
            end = tuple(max_cnt[e][0])
            far = tuple(max_cnt[f][0])
            far_defect.append(far)
            cv2.line(result_frame, start, end, [0, 255, 0], 1)
            cv2.circle(result_frame, far, 2, [0, 0, 255], 3)

    moments = cv2.moments(max_cnt)

    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
        cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
    centerMass = (cx, cy)
    cv2.circle(result_frame, centerMass, 1, [0, 255, 0], 10)

    cv2.imshow('result', result_frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
