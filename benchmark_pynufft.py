import cProfile
import numpy
#import matplotlib.pyplot
import copy

#cm = matplotlib.cm.gray
# load example image
import pkg_resources
from pynufft.pynufft import NUFFT_hsa
from pynufft.pynufft import NUFFT_cpu

DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
import numpy
#import matplotlib.pyplot
import scipy
import scipy.misc

# load example image
#     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
image = scipy.misc.face(gray=True)
image = scipy.misc.imresize(image, (256,256))

image=image.astype(numpy.float)/numpy.max(image[...])
print('loading image...')
#     image[128, 128] = 1.0
Nd = (256, 256)  # image space size
Kd = (512, 512)  # k-space size
Jd = (6, 6)  # interpolation size

# load k-space points
om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
nfft = NUFFT_cpu()  # CPU
nfft.plan(om, Nd, Kd, Jd)

#     nfft.initialize_gpu()
import scipy.sparse
#     scipy.sparse.save_npz('tests/test.npz', nfft.st['p'])
print("create NUFFT gpu object")
NufftObj = NUFFT_hsa()
print("plan nufft on gpu")
NufftObj.plan(om, Nd, Kd, Jd)
NufftObj.offload('ocl')  # for multi-CPU computation
dtype = numpy.complex64

print("NufftObj planed")

y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))
print("send image to device")
NufftObj.x_Nd = NufftObj.thr.to_device( image.astype(dtype))
print("copy image to gx")
gx = NufftObj.thr.copy_array(NufftObj.x_Nd)
print('x close? = ', numpy.allclose(image, NufftObj.x_Nd.get() , atol=1e-4))
NufftObj._x2xx()
print('xx close? = ', numpy.allclose(nfft.x2xx(image), NufftObj.x_Nd.get() , atol=1e-4))

NufftObj._xx2k()

#     print(NufftObj.k_Kd.get(queue=NufftObj.queue, async=True).flags)
#     print(nfft.xx2k(nfft.x2xx(image)).flags)
k = nfft.xx2k(nfft.x2xx(image))
print('k close? = ', numpy.allclose(nfft.xx2k(nfft.x2xx(image)), NufftObj.k_Kd.get() , atol=1e-3*numpy.linalg.norm(k)))

NufftObj._k2y()
NufftObj._y2k()
y2 = NufftObj.y.get()

print('y close? = ', numpy.allclose(y, y2 ,  atol=1e-3*numpy.linalg.norm(y)))
print('k2 close? = ', numpy.allclose(nfft.y2k(y2), NufftObj.k_Kd2.get(), atol=1e-3*numpy.linalg.norm(nfft.y2k(y2)) ))
NufftObj._k2xx()
NufftObj._xx2x()
print('x close? = ', numpy.allclose(nfft.adjoint(y2), NufftObj.x_Nd.get() , atol=1e-3*numpy.linalg.norm(nfft.adjoint(y2))))
image3 = NufftObj.x_Nd.get()

import time
t0 = time.time()
for pp in range(0,10):
	y = nfft.forward(image)
t_cpu = (time.time() - t0)/10.0
print('t_cpu:', t_cpu)

t0 = time.time()
for pp in range(0,10):
	gy = NufftObj.forward(gx)
	y_pynufft = gy.get()
t_gpu = (time.time() - t0)/10.0
print('t_gpu:', t_gpu)
