from scipy.spatial import KDTree
import numpy as np
from math import sin, cos
import cv2


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

            from matplotlib import pyplot
            from mpl_toolkits.mplot3d import Axes3D

            fig = pyplot.figure()
            ax = Axes3D(fig)

            ax.set_label("x - axis")
            ax.set_label("y - axis")
            ax.set_label("z - axis")

            ax.plot(self.pointsA[:, 1], self.pointsA[:, 0], self.pointsA[:, 2], "o",
                    color="#cccccc", ms=4, mew=0.5)
            ax.plot(old_points[:, 1], old_points[:, 0], old_points[:, 2],
                    "o", color="#00cccc", ms=4, mew=0.5)
            ax.plot(self.pointsB[:, 0], self.pointsB[:, 1], self.pointsB[:, 2], "o",
                    color="#ff0000", ms=4, mew=0.5)

            pyplot.show()

        return new_points


def icp_test():
    #point_edges = cv2.imread('build/point_edge.png')
    #image_edges = cv2.imread('build/image_edge.png')
    Y, X = np.mgrid[0:100:5, 0:100:5]
    Z = Y ** 2 + X ** 2
    A = np.vstack([Y.reshape(-1), X.reshape(-1), Z.reshape(-1)]).T
    # Y
    # Z
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


icp_test()
