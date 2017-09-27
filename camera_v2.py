import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def lab_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


cap = cv2.VideoCapture(0)

scale = 3
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(scale * cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(scale * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)

first_frame = None


while True:
    ret, frame = cap.read()

    y1 = int(frame.shape[0] * .05)
    x1 = int(frame.shape[1] * .05)

    y2 = int(frame.shape[0] * .95)
    x2 = int(frame.shape[1] * .5)

    left_up = (x1, y1)
    right_down = (x2, y2)

    # gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)[y1:y2, x1:x2]
    cut = frame[y1:y2, x1:x2]
    # gamma = cv2.medianBlur(adjust_gamma(cut, 1), 7)
    # contrast = cv2.medianBlur(lab_contrast(gamma), 7)
    gray = cv2.medianBlur(cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY), 11)

    if first_frame is None:
        first_frame = gray
        continue

    # delta = knn.apply(gray)

    cv2.rectangle(frame, left_up, right_down, (0, 255, 0), 5)
    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0, 0)

    delta = cv2.absdiff(first_frame, gray)
    cv2.imshow('delta', delta)
    cv2.moveWindow('delta', 1300, 0)

    _, thresh = cv2.threshold(delta, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=3)
    cv2.imshow('thresh', thresh)
    cv2.moveWindow('thresh', 1920, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
