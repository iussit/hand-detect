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

scale = 1.5
first_frames_cnt = 1
contour_area_size = 500
finger_size = 30
min_dist = 25  # >=50 --> нет большого пальца

font = cv2.FONT_HERSHEY_SIMPLEX

first_frames = []
while cap.isOpened():
    _, frame = cap.read()

    if frame is None:
        break

    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    result_frame = frame.copy()

    gamma_frame = median_blur(adjust_gamma(frame, gamma=1))
    contrast_frame = median_blur(lab_contrast(gamma_frame), 5)
    gray_frame = median_blur(cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2GRAY))

    if len(first_frames) < first_frames_cnt:
        first_frames.append(gray_frame)
        continue

    deltas = []
    for first_frame in first_frames:
        deltas.append(cv2.absdiff(first_frame, gray_frame))

    total_delta = None
    for delta in deltas:
        total_delta = delta if total_delta is None else cv2.addWeighted(total_delta, .5, delta, .5, 0)
    total_delta = median_blur(total_delta, 5)

    _, thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=3)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = None
    for contour in contours:
        if cv2.contourArea(contour) < contour_area_size:
            continue
        if max_contour is None:
            max_contour = contour
            continue
        max_contour = np.concatenate((contour, max_contour), axis=0)

    if not (max_contour is None):
        hull_with_points = cv2.convexHull(max_contour)
        hull_without_points = cv2.convexHull(max_contour, returnPoints=False)

        defects = cv2.convexityDefects(max_contour, hull_without_points)

        far_defect = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            far_defect.append(far)
            # cv2.line(result_frame, start, end, (0, 255, 0), 1)
            # cv2.circle(result_frame, far, 1, (100, 255, 255), 10)

        moments = cv2.moments(max_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
        center_mass = (cx, cy)

        dist_betw_defects_to_center = []
        for i in range(0, len(far_defect)):
            dist_betw_defects_to_center.append(
                np.sqrt(np.power(far_defect[i][0] - center_mass[0], 2) +
                        np.power(far_defect[i][1] - center_mass[1], 2))
            )

        aver_defect_dist = np.mean(sorted(dist_betw_defects_to_center)[:2])

        finger = []
        for i in range(0, len(hull_with_points) - 1):
            if (np.absolute(hull_with_points[i][0][0] - hull_with_points[i + 1][0][0]) > finger_size) or (
                        np.absolute(hull_with_points[i][0][1] - hull_with_points[i + 1][0][1]) > finger_size):
                if hull_with_points[i][0][1] >= center_mass[1] - min_dist:
                    finger.append(hull_with_points[i][0])

        fingers = []
        for finger in sorted(finger, key=lambda elem: elem[1])[-5:]:
            finger_dist = np.sqrt(np.power(finger[0] - center_mass[0], 2) + np.power(finger[1] - center_mass[1], 2))
            if finger_dist > aver_defect_dist + min_dist:
                fingers.append(finger)

        # cv2.drawContours(result_frame, [max_contour], -1, (0, 0, 255), 3)
        cv2.fillPoly(result_frame, max_contour, (250, 206, 135))
        cv2.circle(result_frame, center_mass, 2, (100, 0, 255), 20)
        # cv2.putText(frame, 'center', tuple(center_mass), font, 2, (255, 255, 255), 2)
        for finger in fingers:
            cv2.circle(result_frame, tuple(finger), 1, (0, 255, 0), 20)

    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', 0, 0)

    cv2.imshow('result_frame', result_frame)
    cv2.moveWindow('result_frame', 1000, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
