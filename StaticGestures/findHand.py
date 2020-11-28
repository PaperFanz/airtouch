import cv2
import numpy as np
import copy
import math
import json
import io
import os
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image

# parameters
cap_region_x_begin=0.45  # start point/total width
cap_region_y_end=0.6  # start point/total width
threshold = 70  #  BINARY threshold
blurValue = 5  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
count = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
isFistCaptured = 0

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((2, 2), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=2)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing. 

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt+1
    return False, -1

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200) #brightness
camera.set(3,600) #width
camera.set(4,600) #height


while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (0,0),
                   (int(cap_region_x_begin * frame.shape[1]), int(cap_region_y_end * frame.shape[0])), 
                   (255, 0, 0), 2) # draw region of interest
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        # img = img[0:int(cap_region_y_end * frame.shape[0]),
        #         0:int(cap_region_x_begin * frame.shape[1])]  # clip the ROI
        cv2.imshow('removeBG', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0) # blur for better binarization
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # binarize
        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        img = thresh | im_floodfill_inv
        img_bin = img

        cv2.imshow('binary', img_bin)

        hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')
        hand = hand_cascade.detectMultiScale(thresh, 1.3, 5) # DETECTING HAND IN THE THRESHOLDE IMAGE
        mask = np.zeros(thresh.shape, dtype = "uint8") # CREATING MASK
        for (x,y,w,h) in hand: # MARKING THE DETECTED ROI
            cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2) 
            cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)
        thresh = cv2.bitwise_and(thresh, mask)

        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            if count == 14:
                isFinishCal,cnt = calculateFingers(res, drawing)
                if ((cv2.contourArea(hull) - cv2.contourArea(res)) / cv2.contourArea(res)) * 100 < 12:
                    cnt = 0                    
                print("There are this many fingers: " + str(cnt)) # <------------- THE ANSWER
                cv2.imshow('output', drawing)
            
            count = (count + 1) % 15

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
