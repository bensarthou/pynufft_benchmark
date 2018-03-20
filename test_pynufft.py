from __future__ import absolute_import

import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt

from pynufft.pynufft import NUFFT_hsa
from pynufft.pynufft import NUFFT_cpu

import pkg_resources
import sys
import getopt
import time

from utils import convert_locations_to_mask, convert_mask_to_locations


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")


print('PyNUFFT')

Kd = (512, 512)
Jd = (4, 4)
om_path = None
title = 'Pynufft'
gpu = False
adj = True

argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv, "hk:j:o:t:g:a:", ["Kd=", "Jd=", "om_path=", "title=", "gpu=", "adj="])
except getopt.GetoptError:
	print('test.py -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -g <gpu> -a <adjoint>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('test.py -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -g <gpu> -a <adjoint>')
		sys.exit()
	elif opt in ("-k", "--Kd"):
		Kd = (int(arg), int(arg))
	elif opt in ("-j", "--Jd"):
		Jd = (int(arg), int(arg))
	elif opt in ("-o", "--om_path"):
		om_path = arg
	elif opt in ("-t", "--title"):
		title = arg
	elif opt in ("-g", "--gpu"):
		gpu = str2bool(arg)
	elif opt in ("-a", "--adj"):
		adj = str2bool(arg)


# Import image
image = scipy.ndimage.imread('datas/mri_slice.png', mode='L')
image = scipy.misc.imresize(image, (256, 256))
image = image.astype(float) / np.max(image[...])
#
# image = np.ones((1024,1024), dtype=np.float64)
# Kd =(2048, 2048)

# Import non-uniform frequences
try:
	om = np.load('datas/' + om_path)  # between [-0.5, 0.5[
	om = om * (2 * np.pi)
except IOError or AttributeError:
	print('WARNING: Loading NU sample example from Pynufft')
	DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')
	# om is normalized between [-pi, pi]
	om = np.load(DATA_PATH + 'om2D.npz')['arr_0']

Nd = image.shape  # image size

# ## test
# om_ = om/(2*np.pi)
# om_[om_ > (255/256. - 0.5)] = (255/256. - 0.5)
# mask = convert_locations_to_mask(om_, Nd)
# mask_1 = mask[:int(Nd[0]/2),:int(Nd[1]/2)]
# mask_2 = mask[:int(Nd[0]/2),int(Nd[1]/2):]
# mask_3 = mask[int(Nd[0]/2):,:int(Nd[1]/2)]
# mask_4 = mask[int(Nd[0]/2):,int(Nd[1]/2):]
# om_1 = convert_mask_to_locations(mask_1)
# om_2 = convert_mask_to_locations(mask_2)
# om_3 = convert_mask_to_locations(mask_3)
# om_4 = convert_mask_to_locations(mask_4)
#
# plt.figure()
# plt.subplot(421)
# plt.imshow(mask_1, cmap = 'gray')
# plt.subplot(422)
# plt.scatter(om_1[:,0],om_1[:,1], cmap='gray', s= 0.5)
# plt.subplot(423)
# plt.imshow(mask_2, cmap = 'gray')
# plt.subplot(424)
# plt.scatter(om_2[:,0],om_2[:,1], cmap='gray', s= 0.5)
# plt.subplot(425)
# plt.imshow(mask_3, cmap = 'gray')
# plt.subplot(426)
# plt.scatter(om_3[:,0],om_3[:,1], cmap='gray', s= 0.5)
# plt.subplot(427)
# plt.imshow(mask_4, cmap = 'gray')
# plt.subplot(428)
# plt.scatter(om_4[:,0],om_4[:,1], cmap='gray', s= 0.5)
# plt.show()
# exit(0)
# diff_vert = mask[:,:int(Nd[1]/2)] - mask[:,int(Nd[1]/2):]
# diff_hor = mask[:int(Nd[0]/2),:] - mask[int(Nd[0]/2):,:]
#
# diff_diag = mask[:int(Nd[0]/2),:int(Nd[1]/2)] - mask[int(Nd[0]/2):,int(Nd[1]/2):]
# diff_antidiag = mask[:int(Nd[0]/2),int(Nd[1]/2):] - mask[int(Nd[0]/2):,:int(Nd[1]/2)]
# diff_vert_left = mask[:int(Nd[0]/2),:int(Nd[1]/2)] - mask[int(Nd[0]/2):,:int(Nd[1]/2)]
# diff_vert_right = mask[:int(Nd[0]/2),int(Nd[1]/2):] - mask[int(Nd[0]/2):,int(Nd[1]/2):]
#
# print(np.sum(diff_diag))
# print(np.sum(diff_antidiag))
# print(np.sum(diff_vert_left))
# print(np.sum(diff_vert_right))
#
# plt.figure()
# plt.imshow(mask, cmap='gray')
# plt.show()
#
# plt.figure()
# plt.subplot(221)
# plt.imshow(mask[:int(Nd[0]/2),:int(Nd[1]/2)], cmap='gray')
# plt.subplot(222)
# plt.imshow(mask[:int(Nd[0]/2),int(Nd[1]/2):], cmap='gray')
# plt.subplot(223)
# plt.imshow(mask[int(Nd[0]/2):,:int(Nd[1]/2)], cmap='gray')
# plt.subplot(224)
# plt.imshow(mask[int(Nd[0]/2):,int(Nd[1]/2):], cmap='gray')
# plt.show()
#
# plt.figure()
# plt.imshow(mask[:int(Nd[0]/2),:int(Nd[1]/2)] - mask[:int(Nd[0]/2),:int(Nd[1]/2)], cmap='gray')
# plt.show()
# exit(0)


print('setting image dimension Nd...', Nd)
print('setting spectrum dimension Kd...', Kd)
print('setting interpolation size Jd...', Jd)

print('Fourier transform...')
time_pre = time.clock()
# Preprocessing NUFFT
if(gpu == True):
	time_1 = time.clock()
	NufftObj = NUFFT_hsa()
	time_2 = time.clock()
	NufftObj.plan(om, Nd, Kd, Jd)
	time_3 = time.clock()
	# NufftObj.offload('cuda')  # for GPU computation
	NufftObj.offload('ocl')  # for multi-CPU computation
	time_4 = time.clock()
	dtype = np.complex64
	time_5 = time.clock()

	print("send image to device")
	NufftObj.x_Nd = NufftObj.thr.to_device( image.astype(dtype))
	print("copy image to gx")
	time_6 = time.clock()
	gx = NufftObj.thr.copy_array(NufftObj.x_Nd)
	time_7 = time.clock()
	print('total:', time_7 - time_1, '/Decl obj: ', time_2 - time_1, '/plan: ', \
	time_3 - time_2, '/offload: ', time_4 - time_3, '/to_device: ', time_6 - time_5, '\copy_array: ', time_7 - time_6)
else:
	NufftObj = NUFFT_cpu()
	NufftObj.plan(om, Nd, Kd, Jd)


# Compute F_hat
if gpu == True:
	time_comp = time.clock()
	gy = NufftObj.forward(gx)
	y_pynufft = gy.get()
	time_end = time.clock()
else:
	time_comp = time.clock()
	y_pynufft = NufftObj.forward(image)
	time_end = time.clock()

time_preproc = time_comp - time_pre
time_proc = time_end - time_comp
time_total = time_preproc + time_proc

save_pynufft = {'y': y_pynufft, 'Nd': Nd, 'Kd': Kd, 'Jd': Kd, 'om_path': om_path,
				'time_preproc': time_preproc, 'time_proc': time_proc, \
				'time_total': time_total, 'adj':adj, 'title': title}
np.save('datas/'+title+'.npy', save_pynufft)

# Plot K-space
# kx = np.real(y_pynufft)
# ky = np.imag(y_pynufft)
# plt.figure()
# plt.plot(kx,ky, 'w.')
# ax = plt.gca()
# ax.set_facecolor('k')
# plt.title('K-space Pynufft')
# plt.show()
if adj == True:
	# backward test
	print('Self-adjoint Test...')
	if gpu == True:
		gx2 = NufftObj.adjoint(gy)
		img_reconstruct_ = gx2.get()

	else:
		img_reconstruct_ = NufftObj.adjoint(y_pynufft)

	# print(np.abs(img_reconstruct_)/np.max(np.abs(img_reconstruct_)))

	img_reconstruct = np.abs(img_reconstruct_)/np.max(np.abs(img_reconstruct_))
	img_reconstruct = img_reconstruct.astype(np.float64)
	# plt.figure()
	# plt.suptitle('Comparaison original/selfadjoint')
	# plt.subplot(121)
	# plt.imshow(image, cmap='gray')
	# plt.subplot(122)
	# plt.imshow(img_reconstruct, cmap='gray')
	# plt.show()

	save_pynufft = {'y': y_pynufft, 'Nd': Nd, 'Kd': Kd, 'Jd': Kd, 'om_path': om_path,
					'time_preproc': time_preproc, 'time_proc': time_proc,\
					 'time_total': time_total, 'title': title, 'adj':adj,\
					 'img_reconstruct': img_reconstruct, 'img_orig': image}
	np.save('datas/'+title+'.npy', save_pynufft)
