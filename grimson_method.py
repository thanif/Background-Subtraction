import cv2
import os
import numpy as np

def norm_pdf(x,mean,sigma):
    return (1/(np.sqrt(2*3.14)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

mins = []

maxs = []

f_image = cv2.imread("./Imgs/00001.png")

f_image = cv2.cvtColor(f_image, cv2.COLOR_BGR2GRAY)

row,col = f_image.shape

mean = np.zeros([3,row,col],np.float64)
mean[1,:,:] = f_image

variance = np.zeros([3,row,col],np.float64)
variance[:,:,:] = 400

omega = np.zeros([3,row,col],np.float64)
omega[0,:,:],omega[1,:,:],omega[2,:,:] = 0,0,1

omega_by_sigma = np.zeros([3,row,col],np.float64)

foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)

alpha = 0.3
T = 0.5

a = np.uint8([255])
b = np.uint8([0])

for imgs in os.listdir("./Imgs/"):

    c_image = cv2.imread("./Imgs/"+imgs)

    c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)

    c_image = c_image.astype(np.float64)

    variance[0][np.where(variance[0]<1)] = 10
    variance[1][np.where(variance[1]<1)] = 5
    variance[2][np.where(variance[2]<1)] = 1

    sigma1 = np.sqrt(variance[0])
    sigma2 = np.sqrt(variance[1])
    sigma3 = np.sqrt(variance[2])

    compare_val_1 = cv2.absdiff(c_image,mean[0])
    compare_val_2 = cv2.absdiff(c_image,mean[1])
    compare_val_3 = cv2.absdiff(c_image,mean[2])

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3

    fore_index1 = np.where(omega[2]>T)
    fore_index2 = np.where(((omega[2]+omega[1])>T) & (omega[2]<T))

    gauss_fit_index1 = np.where(compare_val_1 <= value1)
    gauss_not_fit_index1 = np.where(compare_val_1 > value1)


    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)

    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)

    temp = np.zeros([row, col])
    temp[fore_index1] = 1
    temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
    index3 = np.where(temp == 2)

    temp = np.zeros([row,col])
    temp[fore_index2] = 1
    index = np.where((compare_val_3<=value3)|(compare_val_2<=value2))
    temp[index] = temp[index]+1
    index2 = np.where(temp==2)

    match_index = np.zeros([row,col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)

    rho = alpha * norm_pdf(c_image[gauss_fit_index1], mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
    constant = rho * ((c_image[gauss_fit_index1] - mean[0][gauss_fit_index1]) ** 2)
    mean[0][gauss_fit_index1] = (1 - rho) * mean[0][gauss_fit_index1] + rho * c_image[gauss_fit_index1]
    variance[0][gauss_fit_index1] = (1 - rho) * variance[0][gauss_fit_index1] + constant
    omega[0][gauss_fit_index1] = (1 - alpha) * omega[0][gauss_fit_index1] + alpha
    omega[0][gauss_not_fit_index1] = (1 - alpha) * omega[0][gauss_not_fit_index1]

    rho = alpha * norm_pdf(c_image[gauss_fit_index2], mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
    constant = rho * ((c_image[gauss_fit_index2] - mean[1][gauss_fit_index2]) ** 2)
    mean[1][gauss_fit_index2] = (1 - rho) * mean[1][gauss_fit_index2] + rho * c_image[gauss_fit_index2]
    variance[1][gauss_fit_index2] = (1 - rho) * variance[1][gauss_fit_index2] + rho * constant
    omega[1][gauss_fit_index2] = (1 - alpha) * omega[1][gauss_fit_index2] + alpha
    omega[1][gauss_not_fit_index2] = (1 - alpha) * omega[1][gauss_not_fit_index2]


    rho = alpha * norm_pdf(c_image[gauss_fit_index3], mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
    constant = rho * ((c_image[gauss_fit_index3] - mean[2][gauss_fit_index3]) ** 2)
    mean[2][gauss_fit_index3] = (1 - rho) * mean[2][gauss_fit_index3] + rho * c_image[gauss_fit_index3]
    variance[2][gauss_fit_index3] = (1 - rho) * variance[2][gauss_fit_index3] + constant
    omega[2][gauss_fit_index3] = (1 - alpha) * omega[2][gauss_fit_index3] + alpha
    omega[2][gauss_not_fit_index3] = (1 - alpha) * omega[2][gauss_not_fit_index3]

    mean[0][not_match_index] = c_image[not_match_index]
    variance[0][not_match_index] = 200
    omega[0][not_match_index] = 0.1

    sum = np.sum(omega,axis=0)
    omega = omega/sum

    omega_by_sigma[0] = omega[0] / sigma1
    omega_by_sigma[1] = omega[1] / sigma2
    omega_by_sigma[2] = omega[2] / sigma3

    index = np.argsort(omega_by_sigma,axis=0)

    mean = np.take_along_axis(mean,index,axis=0)
    variance = np.take_along_axis(variance,index,axis=0)
    omega = np.take_along_axis(omega,index,axis=0)

    c_image = c_image.astype(np.uint8)
    
    background[index2] = c_image[index2]
    background[index3] = c_image[index3]
    #cv2.imshow('frame',cv2.subtract(c_image,background))

    #cv2.waitKey(0)

    cv2.imwrite("./grimson_output/"+"3_"+imgs, cv2.subtract(c_image,background))

