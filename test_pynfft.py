import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt

from pynfft.nfft import NFFT
import pkg_resources
import sys, getopt, time

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

print('PyNFFT')

Kd = (512,512)
Jd = (4,4)
om_path = None
title = 'PyNFFT'
adj = True

argv = sys.argv[1:]
try:
	opts, args = getopt.getopt(argv,"hk:j:o:t:a:",["Kd=","Jd=","om_path=", "title=", "adj="])
except getopt.GetoptError:
	print('test.py -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -a <adjoint>')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print('test.py -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -a <adjoint>')
		sys.exit()
	elif opt in ("-k", "--Kd"):
		Kd = (int(arg), int(arg))
	elif opt in ("-j", "--Jd"):
		Jd = (int(arg), int(arg))
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
print('setting spectrum dimension Kd...', Kd)
print('setting interpolation size Jd...', Jd)

print('Fourier transform...')
## declaration and pre-computation
time_pre = time.clock()
plan = NFFT(image.shape,om.shape[0])
plan.x = om
plan.precompute()
plan.f_hat = image
## Compute 2D-Fourier transform
time_comp = time.clock()
y_nfft = plan.trafo()
time_end = time.clock()

time_preproc = time_comp - time_pre
time_proc = time_end - time_comp
time_total = time_preproc + time_proc

save_pynfft = {'y':y_nfft, 'Nd':Nd, 'Kd':Kd, 'Jd':Kd, 'om_path':om_path,\
 'time_preproc':time_preproc, 'time_proc':time_proc, 'time_total':time_total,\
 'adj':adj, 'title':title}
np.save('datas/'+title+'.npy', save_pynfft)

# ## Plot k-space
# kx = np.real(y_nfft)
# ky = np.imag(y_nfft)
# plt.figure()
# plt.plot(kx,ky, 'w.')
# ax = plt.gca()
# ax.set_facecolor('k')
# plt.title('K-space Pynufft')
# plt.show()

if adj ==True:
	# backward test
	plan.f = y_nfft
	img_reconstruct_ = plan.adjoint()
	img_reconstruct = np.abs(img_reconstruct_)/np.max(np.abs(img_reconstruct_))
	img_reconstruct = img_reconstruct.astype(np.float64)

	# plt.figure()
	# plt.subplot(121)
	# plt.imshow(image, cmap = 'gray')
	# plt.subplot(122)
	# plt.imshow(img_reconstruct, cmap='gray')
	# plt.show()

	save_pynfft = {'y':y_nfft, 'Nd':Nd, 'Kd':Kd, 'Jd':Kd, 'om_path':om_path,\
	 'time_preproc':time_preproc, 'time_proc':time_proc, 'time_total':time_total,\
	  'title':title, 'adj':adj, 'img_reconstruct':img_reconstruct, 'img_orig': image}
	np.save('datas/'+title+'.npy', save_pynfft)
