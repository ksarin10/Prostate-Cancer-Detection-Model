#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:35:55 2023

@author: krishsarin
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def stain_norm(input_image):
    if isinstance(input_image, str):  # Check if the input is a file path
        input_image = cv2.imread(input_image)  # Load the image

    # Convert the input image to a format expected by the function
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    Io = 240
    alpha = 1
    beta = 0.15
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape((-1, 3))
    OD = -np.log10((img.astype(float) + 1) / Io)

    ODhat = OD[~np.any(OD < beta, axis=1)]

    # Regularize the covariance matrix
    epsilon = 1e-5
    cov_matrix = np.cov(ODhat.T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon  # Add a small constant epsilon to the diagonal

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm





