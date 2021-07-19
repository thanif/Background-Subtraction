import cv2
import os
import numpy as np

mins = []

maxs = []

f_image = cv2.imread("./Imgs/00001.png")

f_image = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)

for imgs in os.listdir("./Imgs/"):

    c_image = cv2.imread("./Imgs/"+imgs)

    c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)

    diff = f_image - c_image

    mins.append(np.min(diff))

    maxs.append(np.max(diff))

minimum = np.average(mins)
maximum = np.average(maxs)

threshold = (minimum + maximum)/2

for imgs in os.listdir("./Imgs/"):

    c_image = cv2.imread("./Imgs/"+imgs)

    c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)

    diff = f_image - c_image

    diff[diff>threshold] = 255

    diff[diff<threshold] = 0
   

    #cv2.imshow("Image", diff)

    #cv2.waitKey(0)

    cv2.imwrite("./simple_bs_output/"+"1_"+imgs,diff)
