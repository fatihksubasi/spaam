#!/usr/bin/python

import math
import numpy as np
from scipy import linalg


class SPAAM():
    def __init__(self, pImage, pWorld):
        self.numSamples = len(pImage)
        assert self.numSamples == len(pWorld) and self.numSamples >= 6

        self.pImage = np.asarray(pImage)
        self.pWorld = np.asarray(pWorld)
        self._normalize()
        self._denormalize()

    # private helper method to normalize the points by mean and deviation
    def _normalize(self):
        self.wMean = np.zeros(3)
        self.iMean = np.zeros(2)
        self.wScale = np.zeros(3)
        self.iScale = np.zeros(2)

        for i in range(self.numSamples):
            self.wMean += self.pWorld[i]
            w = np.array([self.pWorld[i, 0] ** 2,
                          self.pWorld[i, 1] ** 2,
                          self.pWorld[i, 2] ** 2, ])
            self.wScale += w

            self.iMean += self.pImage[i]
            t = np.array([self.pImage[i, 0] ** 2,
                          self.pImage[i, 1] ** 2])
            self.iScale += t

        self.wMean /= self.numSamples
        self.wScale /= self.numSamples
        self.iMean /= self.numSamples
        self.iScale /= self.numSamples

        for i in range(3):
            self.wScale[i] = math.sqrt(self.wScale[i] - (self.wMean[i] ** 2))

        for i in range(2):
            self.iScale[i] = math.sqrt(self.iScale[i] - (self.iMean[i] ** 2))

        return 1

    # private helper method to map values back to their original space
    def _denormalize(self):
        self.wDenom = np.zeros((4, 4))
        self.iDenom = np.zeros((3, 3))
        # homogenize the matrices
        self.iDenom[2, 2], self.wDenom[3, 3] = 1., 1.

        for i in range(3):
            self.wDenom[i, i] = 1 / self.wScale[i]
            self.wDenom[i, 3] = -self.wMean[i] * self.wDenom[i, i]

        for i in range(2):
            self.iDenom[i, i] = self.iScale[i]
            self.iDenom[i, 2] = self.iMean[i]

    # Returns positive valued diagonal elements of K Matrix
    # adjusts R accordingly to remain transformation matrix unchanged
    def _correct_diagonal(self, K, R):
        assert K.shape == (3, 3) and R.shape == (3, 3)
        for i in range(3):
            if K[i, i] < 0:
                K[i, i] *= -1
                R[i, i] *= -1

        return K, R

    # computes G matrix
    def get_camera_matrix(self):
        B = np.zeros((2*self.numSamples, 12))

        # decomposing the coefficient matrix B and setting values
        for i in range(self.numSamples):
            # using normalized 2D and 3D points
            image = np.divide((self.pImage[i] - self.iMean), self.iScale)
            world = np.divide((self.pWorld[i] - self.wMean), self.wScale)

            B[i*2][0] = 0
            B[i*2][1] = 0
            B[i*2][2] = 0
            B[i*2][3] = 0
            B[i*2][4] = -world[0]
            B[i*2][5] = -world[1]
            B[i*2][6] = -world[2]
            B[i*2][7] = -1
            B[i*2][8] = image[1] * world[0]
            B[i*2][9] = image[1] * world[1]
            B[i*2][10] = image[1] * world[2]
            B[i*2][11] = image[1]
            B[i*2 + 1][0] = world[0]
            B[i*2 + 1][1] = world[1]
            B[i*2 + 1][2] = world[2]
            B[i*2 + 1][3] = 1
            B[i*2 + 1][4] = 0
            B[i*2 + 1][5] = 0
            B[i*2 + 1][6] = 0
            B[i*2 + 1][7] = 0
            B[i*2 + 1][8] = -image[0] * world[0]
            B[i*2 + 1][9] = -image[0] * world[1]
            B[i*2 + 1][10] = -image[0] * world[2]
            B[i*2 + 1][11] = -image[0]

        # singular value decomposition
        _, _, V = np.linalg.svd(B)
        V = V.transpose()

        # the smallest singular value is generally the last column of V^T
        smallestValues = V[:, 11].reshape(3, 4)

        # denormalizing
        G = np.dot(np.dot(self.iDenom, smallestValues), self.wDenom)

        # normalize matrix in z direction
        normalDirection = math.sqrt(G[2, 0]**2 + G[2, 1]**2 + G[2, 2]**2)
        G *= 1 / normalDirection

        # all projected points should have positive z values
        # check a point and correct G matrix if necessary
        isNegative = G[2, 0] * self.pWorld[0, 0] + G[2, 1] * \
            self.pWorld[0, 1] + G[2, 2] * self.pWorld[0, 2] + G[2, 3]
        if isNegative < 0:
            G *= -1

        return G

     # compute projection and transformation matrices
    def get_transformation_matrix(self, G):
        # decompose G matrix into camera and rotation matrices
        K, R = linalg.rq(G[:, 0:3])

        K, R = self._correct_diagonal(K, R)

        # rotation matrix should have positive determinant by definition
        if np.linalg.det(R) == -1:
            R = -R

        # translation vector
        t = np.dot(np.linalg.inv(K), G[:, 3])

        # composing transformation matrix
        A = np.column_stack((R, t))

        # reshaping camera matrix
        K = np.column_stack((K, np.array([0, 0, 0])))

        return K, A

