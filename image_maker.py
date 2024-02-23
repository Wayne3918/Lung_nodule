from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

img1 = cv2.imread('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\sofa.png',1)
img128 = cv2.resize(img1, (128,128))
dst128 = cv2.cvtColor(img128,cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\128_real.png',dst128)

img64 = cv2.resize(img1, (64,64))
dst64 = cv2.cvtColor(img64,cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\64_real.png',dst64)

img32 = cv2.resize(img1, (32,32))
dst32 = cv2.cvtColor(img32,cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:\\Users\\cheny\\Desktop\\FYP\\Final report images\\project\\32_real.png',dst32)