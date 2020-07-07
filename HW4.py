#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: brianmatejevich
"""

import os, cv2 , numpy as np
hawaiiFolder = "hawaii/"
hawaiiPieces = "hawaii/pieces_aligned/"
hawaiiPiecesRandom = "hawaii/pieces_random/"

#load in a folder of images and put them in a list
def load(folder):
    images = []
    for picture in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,picture))
        if img is not None:
            images.append(img)
    return images

#rotate image by degrees
def rotate(img,degree):
    rows,cols = img.shape[:2]
    imgCenter = (cols//2,rows//2)
    M = cv2.getRotationMatrix2D(imgCenter,degree,1.0)
    imgRotated = cv2.warpAffine(img,M,(cols,rows))
    return imgRotated

def buildPuzzle(original,working_img):
    for piece in imgList:
        # convert to grayscale images
        gray1 = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # SIFT settings
        nFeatures = 0
        nOctaveLayers = 5
        contrastThreshold = .00002  # Threshold to filter out weak features
        edgeThreshold = 13  # Threshold to filter out edges (lower is stricter)
        sigma = 1.3  # The gaussian std dev at octave zero

        # Create SIFT object
        sift = cv2.xfeatures2d.SIFT_create(nFeatures, nOctaveLayers, contrastThreshold,
                                           edgeThreshold, sigma)

        # Detect keypoints and compute their descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des2 is None:
            print('No keypoints found for image 2')
        else:
            # Find matches between keypoints in the two images.
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            if len(matches) == 0:
                print('No matches')
            else:
                matches = sorted(matches, key=lambda x:x.distance)
                num_matches_to_show = 1
                # Loop through the top matches
                for i in range(num_matches_to_show):
                    match = matches[i]

                    curr_kp1 = kp1[match.queryIdx]  # get the keypoint for piece
                    angle1 = curr_kp1.angle
                    loc1 = curr_kp1.pt
                    x1 = int(loc1[0])
                    y1 = int(loc1[1])

                    curr_kp2 = kp2[match.trainIdx]  # get the keypoint for original
                    angle2 = curr_kp2.angle
                    loc2 = curr_kp2.pt
                    x2 = int(loc2[0])
                    y2 = int(loc2[1])
                x = (x2//50)*50
                y = (y2//50)*50

                angle = round(angle1 - angle2)
                if angle < 0:
                    angle = angle + 360
                if angle > 70 and angle < 110:
                    angle = 90
                if angle > 160 and angle < 200:
                    angle = 180
                if angle > 250 and angle < 290:
                    angle = 270
                if angle < 20:
                    angle = 0
                else:
                    angle = angle
                piece = rotate(piece,angle)
                a = working_img[y:y+50,x:x+50]
                working_img[y:y+50,x:x+50] = piece
                cv2.imshow("working..." , working_img)
                cv2.waitKey(10)
    return working_img


imgList = load(hawaiiPieces)
boxList = load(hawaiiFolder)
original = boxList[0]
rows,cols = original.shape[:2]
working_img = cv2.merge((np.zeros([rows,cols]),np.zeros([rows,cols]),np.zeros([rows,cols])))
working_img = working_img.astype(np.uint8)
working_img = buildPuzzle(original,working_img)

cv2.imshow('Box Top', original)
cv2.waitKey(1000)
cv2.destroyAllWindows()

imgList = load(hawaiiPiecesRandom)
working_img = cv2.merge((np.zeros([rows,cols]),np.zeros([rows,cols]),np.zeros([rows,cols])))
working_img = working_img.astype(np.uint8)
working_img = buildPuzzle(original,working_img)
cv2.imshow('Box Top', original)
cv2.waitKey(1000)
cv2.destroyAllWindows()

