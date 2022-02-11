import numpy as np
import cv2
s = cv2.imread("venu.JPG")
# cv2.imshow('',s);cv2.waitKey(0)
# print(np.average(s))
a = s>90
a=a*255
# a=a.astype("uint8")
k = a[:,:,0]
# cv2.imshow('',a);cv2.waitKey(0)

filename = 'savedImage.png'
cv2.imwrite(filename, k)