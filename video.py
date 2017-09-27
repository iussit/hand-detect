import numpy as np
import cv2


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def lab_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def median_blur(image, ksize=3):
    return cv2.medianBlur(image, ksize)


cap = cv2.VideoCapture('input/video.avi')

#15 cadrov
clear_frames = []

while cap.isOpened():
    _, frame = cap.read()

    if frame is None:
        break

    scale = 1.8
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    gamma_frame = median_blur(adjust_gamma(frame, gamma=1.5))
    contrast_frame = median_blur(lab_contrast(gamma_frame), 7)
    gray_frame = median_blur(cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2GRAY))

    if len(clear_frames) < 1:
        clear_frames.append(gray_frame)
        continue

    deltas = []
    for clear_frame in clear_frames:
        deltas.append(cv2.absdiff(clear_frame, gray_frame))

    total_delta = deltas[0]
    for index in range(1, len(deltas)):
        total_delta = cv2.absdiff(total_delta, deltas[index])

    _, thresh = cv2.threshold(total_delta, 30, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=5)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_frame = frame.copy()

    max_cnt = None
    for cnt in contours:
        if max_cnt is None:
            max_cnt = cnt
            continue
        if cv2.contourArea(cnt) > cv2.contourArea(max_cnt):
            max_cnt = cnt

    if cv2.contourArea(max_cnt) > 600:
        cv2.drawContours(result_frame, [max_cnt], 0, (0, 255, 0), 2)

        moments = cv2.moments(max_cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
        centerMass = (cx, cy)
        cv2.circle(result_frame, centerMass, 1, [0, 0, 255], 30)

    cv2.imshow('input', frame)
    cv2.moveWindow('input', 0, 0)
    # cv2.imshow('gray_frame', gray_frame)
    # cv2.imshow('delta', total_delta)
    # cv2.imshow('thresh', thresh)
    cv2.imshow('result', result_frame)
    cv2.moveWindow('result', 1500, 0)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
