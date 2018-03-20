import numpy as np
import scipy.misc
import scipy.ndimage
import sys, time

def progress(count, total, status=''):
	bar_len = 60
	filled_len = int(round(bar_len * count / float(total)))
	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
	sys.stdout.flush()

def ndft_1D(x, f, N):
	"""non-equispaced discrete Fourier transform"""
	k = -(N // 2) + np.arange(N)
	return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))


def ndft_2D(x, f, Nd):
	M,N = Nd[0], Nd[1]
	K = np.shape(x)[0]
	ndft2d = [0.0 for i in range(K)]
	for k in range(K):
		# print('k',k ,'sur ', K)
		progress(k, K)
		sum_ = 0.0
		for m in range(M):
			for n in range(N):
				# print(n,m)
				value = f[m, n]
				e = np.exp(- 1j * 2*np.pi * (x[k,0] + x[k,1]))
				sum_ += value * e
		ndft2d[k] = sum_ / M / N
	return ndft2d

def indft_2d(y, Nd, x):

	res = np.zeros(Nd)
	M,N = Nd[0], Nd[1]
	K = np.shape(x)[0]

	for m in range(M):
		for n in range(N):
			# print(n,m)
			sum_ = 0.0
			for k in range(K):
				e = np.exp(1j * 2*np.pi * (x[k,0] + x[k,1]))
				sum_ += y[k] * e
			pix = int(sum_.real + 0.5)
			res[m, n] = pix
	return res



#Import image
image = scipy.ndimage.imread('datas/mri_slice.png', mode='L')
image = scipy.misc.imresize(image, (256,256))
image= image.astype(float)/np.max(image[...])
# Import non-uniform frequences
om = np.load('datas/om_pynufft.npy') # between [-0.5, 0.5[

print('NDFT')
time_begin = time.clock()
y = ndft_2D(om, image, image.shape)
time_mid = time.clock()
print('INDFT')
image_new = indft_2d(y, image.shape, om)
time_end = time.clock()

print('time NDFT: ', time_mid -time_begin, 'time INDFT', time_end - time_mid)

plt.figure()
plt.subplot(121)
plt.imshow(image, cmap = 'gray')
plt.title('Image originale')
plt.subplot(122)
plt.imshow(image_new, cmap = 'gray')
plt.title('Image après NDFT et INDFT')
np.imsave("datas/ndft_test.png", "PNG")
plt.show()
