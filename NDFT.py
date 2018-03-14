import numpy as np

def ndft_1D(x, f, N):
	"""non-equispaced discrete Fourier transform"""
	k = -(N // 2) + np.arange(N)
	return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))


def ndft_2D(x, f, Nd):
	M,N = Nd[0], Nd[1]
	K = np.shape(x)[0]
	ndft2d = [0.0 for i in range(K)]
	for k in range(np.shape(x)[0]):
		sum_ = 0.0
		for m in range(M):
			for n in range(N):
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
			sum_ = 0.0
			for k in range(K):
				e = np.exp(1j * 2*np.pi * (x[k,0] + x[k,1]))
				sum_ += y[k] * e
			pix = int(sum_.real + 0.5)
			res[m, n] = pix
	return res

# TEST
# Recreate input image from 2D DFT results to compare to input image

#Import image
image = scipy.ndimage.imread('datas/mri_slice.png', mode='L')
image = scipy.misc.imresize(image, (256,256))
image= image.astype(float)/np.max(image[...])
# Import non-uniform frequences
om = np.load('datas/' + om_path) # between [-0.5, 0.5[


image_new = indft_2d(ndft_2D(om, image, image.shape))
plt.figure()
plt.subplot(121)
plt.imshow(image, cmap = 'gray')
plt.title('Image originale')
plt.subplot(122)
plt.imshow(image_new, cmap = 'gray')
plt.title('Image après NDFT et INDFT')
np.imsave("datas/ndft_test.png", "PNG")
