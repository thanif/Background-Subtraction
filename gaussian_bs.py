import cv2
import os
import numpy as np

mins = []

maxs = []

f_image = cv2.imread("./Imgs/00001.png")

f_image = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)

mean = f_image

(col,row) = mean.shape[:2]

var = np.ones((col,row),np.uint8)
var[:col,:row] = 150

alpha = 0.25

for imgs in os.listdir("./Imgs/"):

    c_image = cv2.imread("./Imgs/"+imgs)

    c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)

    new_mean = (1-alpha)*mean + alpha*c_image       
    new_mean = new_mean.astype(np.uint8)
    new_var = (alpha)*(cv2.subtract(c_image,mean)**2) + (1-alpha)*(var)
       
    value  = cv2.absdiff(c_image,mean)
    value = value /np.sqrt(var)
       
        
    mean = np.where(value < 2.5,new_mean,mean)
    var = np.where(value < 2.5,new_var,var)
    a = np.uint8([255])
    b = np.uint8([0])
    background = np.where(value < 2.5,c_image,0)
    forground = np.where(value>=2.5,c_image,b)   
             
    #cv2.imshow('forground',forground)

    #cv2.waitKey(0)

    cv2.imwrite("./gaussian_bs_output/"+"2_"+imgs,forground)

