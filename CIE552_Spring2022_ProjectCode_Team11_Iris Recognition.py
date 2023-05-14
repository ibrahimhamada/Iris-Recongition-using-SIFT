import numpy as np
import cv2
import os
import math
import copy


# This function takes paths of 2 iris images and finds the matching between both of them.
def match_iris(filepath1, filepath2):
    print("Image 1: " + filepath1)
    print("Image 2: " + filepath2)
    print("Extracting Features of Image 1 ...")
    iris_1 = feature_extraction(filepath1)
    print("Extracting Features of Image 2 ...")
    iris_2 = feature_extraction(filepath2)
    print("Feature Extraction is DONE for both images ...")
    getall_matches(iris_1, iris_2, 0.8, 10, 0.15)


# This function takes the filepath and extrcts the iris only from the eye and then finds key points using SIFT.
def feature_extraction(filepath):
    img = load_image(filepath)
    print("Getting Iris Boundaries ...")
    pupil_circle, extrnal_iris_circle = get_iris_boundaries(img)

    print("Equalizing histogram ..")
    roi = get_equalized_iris(img, extrnal_iris_circle, pupil_circle)

    print("Getting roi iris images ...")
    rois = get_rois(roi, pupil_circle, extrnal_iris_circle)

    print("Searching for keypoints ... \n")
    sift = cv2.SIFT_create()
    load_descriptors(sift, rois)

    return rois


# This function takes the eye image and gets Iris Boundries.
def get_iris_boundaries(img):
    pupil_circle = find_pupil(img)  # Finding iris inner boundary (Pupil)
    radius_range = int(math.ceil(pupil_circle[2] * 1.5))  # Finding iris outer boundary radius range
    center_range = int(math.ceil(pupil_circle[2] * 0.25))  # Finding iris outer boundary center range
    extrnal_iris_circle = find_ext_iris(img, pupil_circle, center_range, radius_range)  # Finding iris outer circle
    # Show the obtained iris boundries.
    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    draw_circles(img_1, pupil_circle, extrnal_iris_circle, center_range, radius_range)
    cv2.imshow('iris boundaries', img_1)
    ch = cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pupil_circle, extrnal_iris_circle


# This function gets the edges for a given image provided thresholds T1 and T2 and using Canny Edge Detector.
def get_edges(image, T1=20, T2=100):
    edges = cv2.Canny(image, T1, T2, apertureSize=5)
    edges = cv2.GaussianBlur(edges, (7, 7), 0)
    return edges


# This function calulates the mean circle parameters out of many obtained circles.
def get_mean_circle(circles):
    mean_x = int(np.mean([c[0] for c in circles]))  # Mean of center X-Coordinate
    mean_y = int(np.mean([c[1] for c in circles]))  # Mean of center Y-Coordinate
    mean_r = int(np.mean([c[2] for c in circles]))  # Mean of Radius
    return mean_x, mean_y, mean_r


# This function finds the pupil circle parameters.
def find_pupil(img):
    param1 = 200
    param2 = 120
    c_circles = []
    while (param2 > 35 and len(c_circles) < 100):
        for mdn, thrs in [(m, t) for m in [3, 5, 7] for t in [20, 25, 30, 35, 40, 45, 50, 55, 60]]:
            median = cv2.medianBlur(img, 2 * mdn + 1)  # Median Blur
            ret, thres = cv2.threshold(median, thrs, 255, cv2.THRESH_BINARY_INV)  # Threshold (Binarization)
            edges = get_edges(thres)  # Canny Edges
            # HoughCircles (x, y, radius) = cv2.HoughCircles(image, method, dp, minDist[, param1[, param2[, minRadius[, maxRadius]]]]])
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, np.array([]), param1, param2)  # HoughCircles
            if (circles is not None) and (len(circles) > 0):
                # convert the (x, y) coordinates and radius of the circles
                # to integers
                circles = np.round(circles[0, :]).astype("int")
                for c in circles:
                    c_circles.append(c)
        param2 = param2 - 1
    mean_circle = get_mean_circle(c_circles)
    return mean_circle


# This function finds the Extrnal Circle parameters.
def get_outter_circles(hough_param, median_params, edge_params, pupil_circle, center_range, radius_range, img):
    crt_circles = []
    for mdn, thrs2 in [(m, t) for m in median_params for t in edge_params]:
        median = cv2.medianBlur(img, 2 * mdn + 1)  # Median Blur
        edges = get_edges(median, 0, thrs2)  # Canny Edges
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, np.array([]), 200, hough_param)  # HoughCircles
        if (circles is not None) and (len(circles) > 0):
            # convert the (x, y) coordinates and radius of the
            # circles to integers
            circles = np.round(circles[0, :]).astype("int")
            for (c_col, c_row, r) in circles:
                if point_in_circle(int(pupil_circle[0]), int(pupil_circle[1]), center_range, c_col,
                                   c_row) and r > radius_range:
                    crt_circles.append((c_col, c_row, r))
    return crt_circles


# This function finds the Extrnal Iris Circle parameters.
def find_ext_iris(img, pupil_circle, center_range, radius_range):
    param2 = 120
    total_circles = []
    while (param2 > 40 and len(total_circles) < 50):
        crt_circles = get_outter_circles(param2, [8, 10, 12, 14, 16, 18, 20], [430, 480, 530], pupil_circle,
                                         center_range, radius_range, img)
        if crt_circles:
            total_circles += crt_circles
        param2 = param2 - 1

    if not total_circles:
        return
    return get_mean_circle(total_circles)


# Check whether a pixel or point is in a given circle or not
def point_in_circle(c_col, c_row, c_radius, p_col, p_row):
    return distance(c_col, c_row, p_col, p_row) <= c_radius


# This Function Draws the obtained Circles.
def draw_circles(img, pupil_circle, ext_iris_circle, center_range, radius_range):
    cv2.circle(img, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], (0, 0, 255), 1)  # Pupil circle
    cv2.circle(img, (pupil_circle[0], pupil_circle[1]), 1, (0, 0, 255), 1)  # Center pupil circle
    cv2.circle(img, (pupil_circle[0], pupil_circle[1]), center_range, (0, 255, 255),
               1)  # Extrnal iris center range limit
    cv2.circle(img, (pupil_circle[0], pupil_circle[1]), radius_range, (0, 255, 255),
               1)  # Extrnal iris radius range limit
    cv2.circle(img, (ext_iris_circle[0], ext_iris_circle[1]), ext_iris_circle[2], (0, 255, 0),
               1)  # Draw the outer ext iris circle
    cv2.circle(img, (ext_iris_circle[0], ext_iris_circle[1]), 1, (0, 255, 0), 1)  # Center of extrnal iris circle


def get_equalized_iris(img, ext_iris_circle, pupil_circle):
    mask = img.copy()
    mask[:] = (0)
    # cv2.circle(image, center_coordinates, radius, color, thickness)
    cv2.circle(mask, (ext_iris_circle[0], ext_iris_circle[1]), ext_iris_circle[2], (255), -1)
    cv2.circle(mask, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], (0), -1)
    roi = cv2.bitwise_and(img, mask)

    cv2.imshow('Pre Equalization Iris Region', roi)
    ch = cv2.waitKey(0)
    cv2.destroyAllWindows()

    equ_roi = roi.copy()
    cv2.equalizeHist(roi, equ_roi)
    roi = cv2.addWeighted(roi, 0.0, equ_roi, 1.0, 0)

    cv2.imshow('equalized histogram iris region', roi)
    ch = cv2.waitKey(0)
    cv2.destroyAllWindows()

    return roi


def get_rois(img, pupil_circle, ext_circle):
    bg = img.copy()
    bg[:] = 0

    init_dict = {'img': bg.copy(),
                 'pupil_circle': pupil_circle,
                 'ext_circle': ext_circle,
                 'kp': None,
                 'des': None
                 }

    rois = copy.deepcopy(init_dict)

    for p_col in range(img.shape[1]):
        for p_row in range(img.shape[0]):
            if not point_in_circle(pupil_circle[0], pupil_circle[1],
                                   pupil_circle[2], p_col, p_row) and \
                    point_in_circle(ext_circle[0], ext_circle[1], ext_circle[2],
                                    p_col, p_row):
                rois['img'][p_row, p_col] = img[p_row, p_col]

    rois['ext_circle'] = \
        (int(1 * ext_circle[2]),
         int(1 * ext_circle[2]),
         int(ext_circle[2]))

    tx = rois['ext_circle'][0] - ext_circle[0]
    ty = rois['ext_circle'][1] - ext_circle[1]
    rois['pupil_circle'] = (int(tx + pupil_circle[0]),
                            int(ty + pupil_circle[1]),
                            int(pupil_circle[2]))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    rois['img'] = cv2.warpAffine(
        rois['img'], M,
        (img.shape[1], img.shape[0]))

    rois['img'] = \
        rois['img'][0:2 * ext_circle[2], 0:2 * ext_circle[2]]

    return rois


def load_descriptors(sift, rois):
    rois['kp'], rois['des'] = \
        sift.detectAndCompute(rois['img'], rois['kp'])



def get_matches(roipos_1, roipos_2,
                dratio, stdev_angle, stdev_dist):
    if not roipos_1['kp'] or not roipos_2['kp']:
        print("KeyPoints not found in one of roipos_x['kp'] !!!")
        return []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(roipos_1['des'], roipos_2['des'], k=2)
    kp1 = roipos_1['kp']
    kp2 = roipos_2['kp']

    diff_dist_1 = roipos_1['ext_circle'][2] - roipos_1['pupil_circle'][2]
    diff_dist_2 = roipos_2['ext_circle'][2] - roipos_2['pupil_circle'][2]

    diff_angles = []
    diff_dists = []
    filtered = []
    for m, n in matches:
        if (m.distance / n.distance) > dratio:
            continue

        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt

        angle_1 = angle_v(
            x1, y1,
            roipos_1['pupil_circle'][0],
            roipos_1['pupil_circle'][1])
        angle_2 = angle_v(
            x2, y2,
            roipos_2['pupil_circle'][0],
            roipos_2['pupil_circle'][1])
        diff_angle = angle_1 - angle_2
        diff_angles.append(diff_angle)

        dist_1 = distance(x1, y1,
                          roipos_1['pupil_circle'][0],
                          roipos_1['pupil_circle'][1])
        dist_1 = dist_1 - roipos_1['pupil_circle'][2]
        dist_1 = dist_1 / diff_dist_1

        dist_2 = distance(x2, y2,
                          roipos_2['pupil_circle'][0],
                          roipos_2['pupil_circle'][1])
        dist_2 = dist_2 - roipos_2['pupil_circle'][2]
        dist_2 = dist_2 / diff_dist_2

        diff_dist = dist_1 - dist_2
        diff_dists.append(diff_dist)

        filtered.append(m)

    # Remove bad matches
    if True and filtered:
        median_diff_angle = median(diff_angles)
        median_diff_dist = median(diff_dists)
        # print "median dist:", median_diff_dist
        for m in filtered[:]:
            x1, y1 = kp1[m.queryIdx].pt
            x2, y2 = kp2[m.trainIdx].pt

            angle_1 = angle_v(
                x1, y1,
                roipos_1['pupil_circle'][0],
                roipos_1['pupil_circle'][1])
            angle_2 = angle_v(
                x2, y2,
                roipos_2['pupil_circle'][0],
                roipos_2['pupil_circle'][1])
            diff_angle = angle_1 - angle_2

            good_diff_angle = \
                (diff_angle > median_diff_angle - stdev_angle and \
                 diff_angle < median_diff_angle + stdev_angle)

            dist_1 = distance(x1, y1,
                              roipos_1['pupil_circle'][0],
                              roipos_1['pupil_circle'][1])
            dist_1 = dist_1 - roipos_1['pupil_circle'][2]
            dist_1 = dist_1 / diff_dist_1

            dist_2 = distance(x2, y2,
                              roipos_2['pupil_circle'][0],
                              roipos_2['pupil_circle'][1])
            dist_2 = dist_2 - roipos_2['pupil_circle'][2]
            dist_2 = dist_2 / diff_dist_2

            diff_dist = dist_1 - dist_2
            good_dist = (diff_dist > median_diff_dist - stdev_dist and \
                         diff_dist < median_diff_dist + stdev_dist)

            if good_diff_angle and good_dist:
                continue

            filtered.remove(m)

    return filtered

def getall_matches(rois_1, rois_2, dratio,
                   stdev_angle, stdev_dist):
    img_matches = []
    numberof_matches = 0
    if not rois_1['kp'] or not rois_2['kp']:
        print("KeyPoints not found in one of rois_x[pos]['kp'] !!!")
        print(" -->" , len(rois_1['kp']), len(rois_2['kp']))
    else:
        matches = get_matches(rois_1, rois_2, dratio, stdev_angle, stdev_dist)
        numberof_matches = len(matches)

        print("{0} matches: {1}".format('complete iris', str(len(matches))))
        crt_image = cv2.drawMatchesKnn(
        rois_1['img'], rois_1['kp'],
        rois_2['img'], rois_2['kp'],
        [matches], flags=2, outImg=None)

        img_matches.append(crt_image)
        cv2.imshow('matches', crt_image)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return numberof_matches

def angle_v(x1, y1, x2, y2):
    return math.degrees(math.atan2(-(y2 - y1), (x2 - x1)))


def distance(x1, y1, x2, y2):
    dst = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dst


def median(x):
    return np.median(np.array(x))


def load_image(filepath):
    img = cv2.imread(filepath, 0)
    cv2.imshow(filepath, img)
    ch = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


if __name__ == "__main__":

    # Specify 2 image paths
    filepath1 = r'./Data/1/Img_1_2_1.jpg'
    filepath2 = r'./Data/1/Img_1_2_3.jpg'

    if os.path.isfile(filepath1) and os.path.isfile(filepath2):
        match_iris(filepath1, filepath2)