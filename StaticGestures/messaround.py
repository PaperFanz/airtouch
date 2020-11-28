# IMPORTING LIBRARIES
import numpy as np
import cv2
import math

#LOADING HAND CASCADE
hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')

# VIDEO CAPTURE
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # blur for better binarization
    cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # binarize
    im_floodfill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    img = thresh | im_floodfill_inv
    img_bin = img
    
    hand = hand_cascade.detectMultiScale(thresh, 1.3, 5) # DETECTING HAND IN THE THRESHOLDE IMAGE
    mask = np.zeros(thresh.shape, dtype = "uint8") # CREATING MASK
    for (x,y,w,h) in hand: # MARKING THE DETECTED ROI
	    cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2) 
	    cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)
    img2 = cv2.bitwise_and(thresh, mask)
    final = cv2.GaussianBlur(img2,(7,7),0)
    contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, 0, (255,255,0), 3)
    cv2.drawContours(final, contours, 0, (255,255,0), 3)

    if len(contours) > 0:
        cnt=contours[0]
        hull = cv2.convexHull(cnt, returnPoints=False)
        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defect
        if defects!= None:
            for i in range(defects.shape[0]):
                p,q,r,s = defects[i,0]
                finger1 = tuple(cnt[p][0])
                finger2 = tuple(cnt[q][0])
                dip = tuple(cnt[r][0])
                # find length of all sides of triangle
                a = math.sqrt((finger2[0] - finger1[0])**2 + (finger2[1] - finger1[1])**2)
                b = math.sqrt((dip[0] - finger1[0])**2 + (dip[1] - finger1[1])**2)
                c = math.sqrt((finger2[0] - dip[0])**2 + (finger2[1] - dip[1])**2)
                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57.29
                # ignore angles > 90 and highlight rest with red dots
                if angle <= 90:
                    count_defects += 1
        # define actions required
        if count_defects == 1:
            cv2.putText(img,"THIS IS 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 2:
            cv2.putText(img, "THIS IS 3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 3:
            cv2.putText(img,"This is 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img,"THIS IS 5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    cv2.imshow('img',thresh)
    cv2.imshow('img1',img)
    cv2.imshow('img2',img2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()