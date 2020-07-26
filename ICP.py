from scipy.spatial import KDTree
import numpy as np
from math import sin, cos
import cv2

width = 938
height = 606


def calcRigidTransformation(MatA, MatB):
    A, B = np.copy(MatA), np.copy(MatB)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A = A - centroid_A
    B = B - centroid_B

    H = np.dot(A.T, B)
    U, S, V = np.linalg.svd(H)
    R = np.dot(V.T, U.T)
    T = np.dot(-R, centroid_A) + centroid_B

    return R, T


class ICP(object):
    def __init__(self, pointsA, pointsB):
        self.pointsA = pointsA
        self.pointsB = pointsB
        self.kdtree = KDTree(self.pointsA)

    def claculate(self, iter):
        old_points = np.copy(self.pointsB)
        new_points = np.copy(self.pointsB)

        for i in range(iter):
            neighbor_idx = self.kdtree.query(old_points)[1]
            targets = self.pointsA[neighbor_idx]
            R, T = calcRigidTransformation(old_points, targets)
            new_points = np.dot(R, old_points.T).T + T

            if np.sum(np.abs(old_points - new_points)) < 0.000000001:
                break

            old_points = new_points

            current = np.zeros((height, width, 3), np.uint8)
            for j in range(self.pointsA)

        return new_points


def icp_test():
    point_edges = cv2.imread('build/point_edge.png')
    image_edges = cv2.imread('build/image_edge.png')
    point_xs = []
    point_ys = []
    image_xs = []
    image_ys = []

    for i in range(height):
        for j in range(width):
            if point_edges[i][j][0] > 0:
                point_xs.append(j)
                point_ys.append(i)

            if image_edges[i][j][0] > 0:
                image_xs.append(i)
                image_ys.append(j)

    print(len(point_ys))
    print(len(image_xs))

    point_X = np.array(point_xs)
    point_Y = np.array(point_ys)
    image_X = np.array(image_xs)
    image_Y = np.array(image_ys)

    A = np.vstack([point_X.reshape(-1), point_Y.reshape(-1)]).T
    B = np.vstack([image_X.reshape(-1), image_Y.reshape(-1)]).T

    R = np.array([
        [1, 0]
        [0, 1]
    ])

    T = np.array([0, 0])

    icp = ICP(A, B)
    points = icp.claculate(3000)

    """

    Y, X = np.mgrid[0:100:5, 0:100:5]
    Z = Y ** 2 + X ** 2
    A = np.vstack([Y.reshape(-1), X.reshape(-1), Z.reshape(-1)]).T
    # Y
    # X
    # Z

    R = np.array([
        [cos(50), -sin(50), 0],
        [sin(50), cos(50), 0],
        [0, 0, 1]
    ])

    T = np.array([5.0, 20.0, 10.0])
    B = np.dot(R, A.T).T + T

    icp = ICP(A, B)
    points = icp.claculate(3000)
    """


icp_test()
