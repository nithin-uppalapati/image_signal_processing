import numpy as np
import cv2

s = cv2.imread('cells_scale.png')
target = np.zeros(s.shape)

# scale s by 0.8 and 1.3
scl = 0.8

h = s.shape[0]
w = s.shape[1]

ka = (h-1)/2
kb = (w-1)/2
a=np.linspace(-ka,ka,h)
b=np.linspace(-kb,kb,w)

x,y = np.meshgrid(b,a)

temp = np.zeros([h,w,2])
temp[:,:,0] = y # x direction
temp[:,:,1] = x # y direction

scl_temp = temp/scl

scl_temp[:,:,0] = scl_temp[:,:,0] + ka
scl_temp[:,:,1] = scl_temp[:,:,1] + kb

x_margin = scl_temp[:,:,0] - np.floor(scl_temp[:,:,0]) # a and b in bilinear interpolation
y_margin = scl_temp[:,:,1] - np.floor(scl_temp[:,:,1]) # a and b in bilinear interpolation

holes_mtx = np.ones(x_margin.shape)

holes_mtx[np.where(scl_temp[:,:,0]<0)] = 0
holes_mtx[np.where(scl_temp[:,:,1]<0)] = 0
holes_mtx[np.where(scl_temp[:,:,0]>h-1)] = h-1
holes_mtx[np.where(scl_temp[:,:,1]>w-1)] = w-1




# print(scl_temp)
int_scl_temp = scl_temp.copy()
int_scl_temp[np.where(scl_temp[:,:,0]<0)]=0
int_scl_temp[np.where(scl_temp[:,:,1]<0)]=0
int_scl_temp[np.where(scl_temp[:,:,0]>h-1)]=h-1
int_scl_temp[np.where(scl_temp[:,:,1]>w-1)]=w-1

print(scl_temp.shape)
col = 0
for col in [0,1,2]:   # rgb channels
	for i in range(4):# 4 steps in bilinear transformation addition
		if i==0:
			int_xs = np.floor(int_scl_temp[:,:,0]).astype(int)
			int_ys = np.floor(int_scl_temp[:,:,1]).astype(int)
			tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
			tp_variable = tp_variable*(1-x_margin)*(1-y_margin)
			target[:,:,col] = target[:,:,col] + tp_variable
			target[:,:,col] = target[:,:,col]*holes_mtx
			print("in i=0")
		if i==1:
			int_xs = np.floor(int_scl_temp[:,:,0]).astype(int)
			int_ys = np.ceil(int_scl_temp[:,:,1]).astype(int)
			tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
			tp_variable = tp_variable*(1-x_margin)*(y_margin)
			target[:,:,col] = target[:,:,col] + tp_variable
			target[:,:,col] = target[:,:,col]*holes_mtx
			print("in i=1")
		if i==2:
			int_xs = np.ceil(int_scl_temp[:,:,0]).astype(int)
			int_ys = np.floor(int_scl_temp[:,:,1]).astype(int)
			tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
			tp_variable = tp_variable*(x_margin)*(1-y_margin)
			target[:,:,col] = target[:,:,col] + tp_variable
			target[:,:,col] = target[:,:,col]*holes_mtx
			print("in i=2")
		if i==3:
			int_xs = np.ceil(int_scl_temp[:,:,0]).astype(int)
			int_ys = np.ceil(int_scl_temp[:,:,1]).astype(int)
			tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
			tp_variable = tp_variable*(x_margin)*(y_margin)
			target[:,:,col] = target[:,:,col] + tp_variable
			target[:,:,col] = target[:,:,col]*holes_mtx
			print("in i=3")

print(target.shape)
filename = 'lab_res_cells.png'
cv2.imwrite(filename, target.astype("uint8"))