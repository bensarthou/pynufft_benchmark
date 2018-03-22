import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt

from NDFT_c import *

import pkg_resources
import sys, getopt, time

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

print('NDFT')

om_path = None
title = 'NDFT'
adj = True

argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv,"ho:t:a:",["om_path=", "title=", "adj="])
except getopt.GetoptError:
	print('test_ndft.py -o <omPath> -t <title> -a <adjoint>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('test.py -o <omPath> -t <title>')
		sys.exit()
	elif opt in ("-o", "--om_path"):
		om_path = arg
	elif opt in ("-t", "--title"):
		title = arg
	elif opt in ("-a", "--adj"):
		adj = str2bool(arg)


#Import image
image = scipy.ndimage.imread('datas/mri_slice.png', mode='L')
image = scipy.misc.imresize(image, (256,256))
image= image.astype(float)/np.max(image[...])

## Import non-uniform frequences
try:
	om = np.load('datas/' + om_path) # between [-0.5, 0.5[
except IOError:
	print('WARNING: Loading NU sample example from Pynufft')
	DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')
	# om is normalized between [-pi, pi] but should be in [-0.5, 0.5[
	om = np.load(DATA_PATH+'om2D.npz')['arr_0']
	om = om/(2*np.pi)

Nd = image.shape  # image size

print('setting image dimension Nd...', Nd)

print('Fourier transform...')
## declaration and pre-computation
time_pre = time.clock()
## Compute 2D-Fourier transform
time_comp = time.clock()
y_ndft = ndft_2D(om, image, Nd)
time_end = time.clock()

time_preproc = time_comp - time_pre
time_proc = time_end - time_comp
time_total = time_preproc + time_proc

save_pyndft = {'y':y_ndft, 'Nd':Nd, 'om_path':om_path,\
 'time_preproc':time_preproc, 'time_proc':time_proc, 'time_total':time_total, 'title':title}
np.save('datas/'+title+'.npy', save_pyndft)

# ## Plot k-space
# kx = np.real(y_nfft)
# ky = np.imag(y_nfft)
# plt.figure()
# plt.plot(kx,ky, 'w.')
# ax = plt.gca()
# ax.set_facecolor('k')
# plt.title('K-space Pynufft')
# plt.show()


if adj == True:
	# backward test
	print('Self-adjoint Test...')

	img_reconstruct_ = indft_2d(y_ndft,Nd,om)

	img_reconstruct = np.abs(img_reconstruct_)/np.max(np.abs(img_reconstruct_))
	img_reconstruct = img_reconstruct.astype(np.float64)
	# plt.figure()
	# plt.suptitle('Comparaison original/selfadjoint')
	# plt.subplot(121)
	# plt.imshow(image, cmap='gray')
	# plt.subplot(122)
	# plt.imshow(img_reconstruct, cmap='gray')
	# plt.show()

	save_pyndft = {'y': y_ndft, 'Nd': Nd, 'om_path': om_path,
					'time_preproc': time_preproc, 'time_proc': time_proc,\
					 'time_total': time_total, 'title': title, 'adj':adj,\
					 'img_reconstruct': img_reconstruct, 'img_orig': image}
	np.save('datas/'+title+'.npy', save_pyndft)
