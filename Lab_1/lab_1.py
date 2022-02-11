import numpy as np
import cv2

def scale(scl, s):
	ka = (h-1)/2; a=np.linspace(-ka,ka,h)
	kb = (w-1)/2; b=np.linspace(-kb,kb,w)

	x,y = np.meshgrid(b,a)

	temp = np.zeros([h,w,2])
	temp[:,:,0] = y # x direction
	temp[:,:,1] = x # y direction
	scl_temp = temp/scl		# #inverse map, hence 1/scl

	scl_temp[:,:,0] = scl_temp[:,:,0] + ka
	scl_temp[:,:,1] = scl_temp[:,:,1] + kb

	x_margin = scl_temp[:,:,0] - np.floor(scl_temp[:,:,0]) # a and b in bilinear interpolation
	y_margin = scl_temp[:,:,1] - np.floor(scl_temp[:,:,1]) # a and b in bilinear interpolation

	target = bilinear_transofrm(s, scl_temp, x_margin, y_margin)
	return target


def chk_indxs(x_arrs, y_arrs):
	x_arrs[np.where(x_arrs<0)]=0
	y_arrs[np.where(y_arrs<0)]=0
	x_arrs[np.where(x_arrs>h-1)]=h-1
	y_arrs[np.where(y_arrs>w-1)]=w-1
#
def bilinear_transofrm(s, int_scl_temp, x_margin, y_margin):
	holes_mtx = np.ones(x_margin.shape)
	holes_mtx[np.where(int_scl_temp[:,:,0]<0)] = 0
	holes_mtx[np.where(int_scl_temp[:,:,1]<0)] = 0
	holes_mtx[np.where(int_scl_temp[:,:,0]>h-1)] = 0
	holes_mtx[np.where(int_scl_temp[:,:,1]>w-1)] = 0

	target = np.zeros(s.shape)
	for col in [0,1,2]:   # rgb channels
		for i in range(4):# 4 sum terms in bilinear transformation addition
			if i==0:
				int_xs = np.floor(int_scl_temp[:,:,0]).astype(int)
				int_ys = np.floor(int_scl_temp[:,:,1]).astype(int)
				chk_indxs(int_xs,int_ys)
				tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
				tp_variable = tp_variable*(1-x_margin)*(1-y_margin)
				target[:,:,col] = target[:,:,col] + tp_variable
				target[:,:,col] = target[:,:,col]*holes_mtx
				# print("in i=0")
			if i==1:
				int_xs = np.floor(int_scl_temp[:,:,0]).astype(int)
				int_ys = np.ceil(int_scl_temp[:,:,1]).astype(int)
				chk_indxs(int_xs,int_ys)
				tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
				tp_variable = tp_variable*(1-x_margin)*(y_margin)
				target[:,:,col] = target[:,:,col] + tp_variable
				target[:,:,col] = target[:,:,col]*holes_mtx
				# print("in i=1")
			if i==2:
				int_xs = np.ceil(int_scl_temp[:,:,0]).astype(int)
				int_ys = np.floor(int_scl_temp[:,:,1]).astype(int)
				chk_indxs(int_xs,int_ys)
				tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
				tp_variable = tp_variable*(x_margin)*(1-y_margin)
				target[:,:,col] = target[:,:,col] + tp_variable
				target[:,:,col] = target[:,:,col]*holes_mtx
				# print("in i=2")
			if i==3:
				int_xs = np.ceil(int_scl_temp[:,:,0]).astype(int)
				int_ys = np.ceil(int_scl_temp[:,:,1]).astype(int)
				chk_indxs(int_xs,int_ys)
				tp_variable = s[list(int_xs.flatten()),list(int_ys.flatten())][:,col].reshape(h,w)
				tp_variable = tp_variable*(x_margin)*(y_margin)
				target[:,:,col] = target[:,:,col] + tp_variable
				target[:,:,col] = target[:,:,col]*holes_mtx
				# print("in i=3")
		print("single color done")
	return target


def rotate(deg, s):
	deg_r = deg*np.pi/180
	ka = (h-1)/2; a=np.linspace(-ka,ka,h)
	kb = (w-1)/2; b=np.linspace(-kb,kb,w)
	x,y = np.meshgrid(b,a)
	temp = np.zeros([h,w,2])
	temp[:,:,0] = y # x direction
	temp[:,:,1] = x # y direction
	scl_temp = temp.reshape(h*w,2)
	R = np.array([[np.cos(deg_r), np.sin(deg_r)],[-1*np.sin(deg_r), np.cos(deg_r)]]) # -deg, because of inverse mapping
	scl_temp = (R@scl_temp.T).T
	scl_temp = scl_temp.reshape(h,w,2)
	scl_temp[:,:,0] = scl_temp[:,:,0] + ka
	scl_temp[:,:,1] = scl_temp[:,:,1] + kb

	x_margin = scl_temp[:,:,0] - np.floor(scl_temp[:,:,0]) # a and b in bilinear interpolation
	y_margin = scl_temp[:,:,1] - np.floor(scl_temp[:,:,1]) # a and b in bilinear interpolation

	target = bilinear_transofrm(s, scl_temp, x_margin, y_margin)
	return target


def trans(amount, s):
	ka = (h-1)/2; a=np.linspace(-ka,ka,h)
	kb = (w-1)/2; b=np.linspace(-kb,kb,w)
	x,y = np.meshgrid(b,a)
	temp = np.zeros([h,w,2])
	temp[:,:,0] = y # x direction
	temp[:,:,1] = x # y direction
	scl_temp = temp

	scl_temp[:,:,0] = scl_temp[:,:,0] + ka - amount[0] #inverse map, hence -ve
	scl_temp[:,:,1] = scl_temp[:,:,1] + kb - amount[1]

	x_margin = scl_temp[:,:,0] - np.floor(scl_temp[:,:,0]) # a and b in bilinear interpolation
	y_margin = scl_temp[:,:,1] - np.floor(scl_temp[:,:,1]) # a and b in bilinear interpolation

	target = bilinear_transofrm(s, scl_temp, x_margin, y_margin)
	return target
##########################################



source = cv2.imread('pisa_rotate.png')
h = source.shape[0]
w = source.shape[1]
# s = cv2.imread('pisa_rotate.png')

# scale s by 0.8 and 1.3
scl = 1.3
degrees = 4
amount = [3.75, 4.3]

# y = scale(scl, source)
# y = rotate(degrees, source)
y = trans(amount, source)

# call the above functions with approproate arguments to do appropriate transformations

filename = 'answer.png'
cv2.imwrite(filename, y.astype("uint8"))