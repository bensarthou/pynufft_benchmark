from __future__ import absolute_import

import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt

# from pynufft.pynufft import NUFFT_hsa
# from pynufft.pynufft import NUFFT_cpu

from pynufft import NUFFT_hsa
from pynufft import NUFFT_cpu

import pkg_resources
import sys
import getopt
import time

from utils_3D import convert_locations_to_mask, convert_mask_to_locations

from memory_profiler import memory_usage

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")


print('PyNUFFT')

Kd1 = 512
Jd1 = 4
om_path = 'samples_2D.npy'
title = 'Pynufft'
gpu = False
adj = True
imgPath = 'mri_img_2D.npy'

argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv, "hi:k:j:o:t:g:a:", ["img=","Kd=", "Jd=", "om_path=", "title=", "gpu=", "adj="])
except getopt.GetoptError:
	print('test_pynufft.py -i <image> -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -g <gpu> -a <adjoint>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('test_pynufft.py -i <image> -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -g <gpu> -a <adjoint>')
		sys.exit()
	elif opt in ("-i", "--img"):
		imgPath = arg
	elif opt in ("-k", "--Kd"):
		Kd1 = int(arg)
	elif opt in ("-j", "--Jd"):
		Jd1 = int(arg)
	elif opt in ("-o", "--om_path"):
		om_path = arg
	elif opt in ("-t", "--title"):
		title = arg
	elif opt in ("-g", "--gpu"):
		gpu = str2bool(arg)
	elif opt in ("-a", "--adj"):
		adj = str2bool(arg)


### Import image
# image = scipy.ndimage.imread('datas/mri_slice.png', mode='L')
# image = scipy.misc.imresize(image, (256,256))
# image = image.astype(float) / np.max(image[...])
# print(image)
image = np.load('datas/'+imgPath)
# image = image[224:288,224:288,224:288]
Nd = image.shape  # image size

# Detection of dimension:
dim = len(Nd)

# Import non-uniform frequences
try:
	om = np.load('datas/' + om_path)  # between [-0.5, 0.5[
	om = om * (2 * np.pi) # because Pynufft want it btw [-pi, pi[
except IOError or AttributeError:
	print('WARNING: Loading NU sample example from Pynufft')
	DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')
	# om is normalized between [-pi, pi]
	om = np.load(DATA_PATH + 'om2D.npz')['arr_0']

assert(om.shape[1]==dim)
Kd = (Kd1,)*dim
Jd =(Jd1,)*dim


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
	# mem_usage =  memory_usage((NufftObj.plan,(om, Nd, Kd, Jd)))
	# print(mem_usage)

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
	# mem_usage = memory_usage((NufftObj.plan,(om, Nd, Kd, Jd)))
	# print(mem_usage)
	NufftObj.plan(om, Nd, Kd, Jd)


# Compute F_hat
if gpu == True:
	time_comp = time.clock()
	gy = NufftObj.forward(gx)
	# print(type(gy))
	# gy = np.array(gy)
	# print(type(gy))
	# print(gy)
	# exit(0)
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
