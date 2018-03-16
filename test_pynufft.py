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

print('setting image dimension Nd...', Nd)
print('setting spectrum dimension Kd...', Kd)
print('setting interpolation size Jd...', Jd)

print('Fourier transform...')
time_pre = time.clock()
# Preprocessing NUFFT
if(gpu == True):
	NufftObj = NUFFT_hsa()
	NufftObj.offload('cuda')  # for GPU computation
else:
	NufftObj = NUFFT_cpu()

NufftObj.plan(om, Nd, Kd, Jd)

# Compute F_hat
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
	img_reconstruct_ = NufftObj.adjoint(y_pynufft)
	img_reconstruct = np.abs(img_reconstruct_)/np.max(np.abs(img_reconstruct_))
	img_reconstruct = img_reconstruct.astype(np.float64)

	# plt.figure()
	# plt.suptitle('Comparaison original/selfadjoint')
	# plt.subplot(121)
	# plt.imshow(image, cmap='gray')
	# plt.subplot(122)
	# plt.imshow(img_reconstruct, cmap='gray')
	# plt.show()

	print(img_reconstruct.dtype)
	save_pynufft = {'y': y_pynufft, 'Nd': Nd, 'Kd': Kd, 'Jd': Kd, 'om_path': om_path,
					'time_preproc': time_preproc, 'time_proc': time_proc,\
					 'time_total': time_total, 'title': title, 'adj':adj,\
					 'img_reconstruct': img_reconstruct, 'img_orig': image}
	np.save('datas/'+title+'.npy', save_pynufft)
