# PyNufft benchmarking

## I - Installation

PyNufft can be installed through pip (`pip3 install pynufft`) or via the project Github.

For CPU computation, it only uses Numpy with pretty great accuracy (but slow computation time).

In order to use GPU/multi-CPU acceleration, you will need the following Python modules:

* reikna (needs pyopencl and pycuda)
* pyopencl (needs a valid installation of OpenCL on your machine)
* pycuda (needs CUDA-compatible GPU and valid installation of CUDA Toolkit on your machine)

### Issues

Installing CUDA and OpenCL on a Ubuntu machine can be really tricky and unstable.

* [CUDA installation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html): Be sure to have a compatible GPU, updated drivers and to follow **every** step
* [OpenCL installation](https://doc.ubuntu-fr.org/opencl): OpenCL works on "plateforms", one for GPU, one for multi-CPU, one for both. While installing CUDA platform seems feasible on Ubuntu 16.04, CPU-Intel plateform must be done with `beignet` package, which isn't stable.

The unstability of those packages put in jeopardy the easy installation and use of GPU Nufft on Ubuntu 16.04

You can find in the project archive/git two python scripts `benchmark_pycda.py` and `benchmark_pyopencl.py` that can be run to check installation of pyopencl and pycuda. There is also a `diagnose()` function in Pynufft to test the availability of the plateform:

```python
    from pynufft.src._helper.helper import diagnose
    diagnose()
```

## II- Tutoriel of benchmarking scripts


Examples on how to call NUFFT on 1D and 2D arrays can be found in PyNufft module. These are working perfectly fine, especially if you only use CPU computation.
One goal here was to benchmark the performance of Pynufft versus other Python Non Uniform Fourier Transform modules, such as pynfft, which uses NFFT/FFTW binding to perform the non-uniform fast transform.

### Code tutoriel
I have written some Python and bash scripts to help compare results and performances of both Python modules. The 4 main scripts are:

* `test_pynufft.py -i <image> -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -g <gpu> -a <adjoint>`
* `test_pynfft.py -k <kspaceSize> -j <kernelSize> -o <omPath> -t <title> -a <adjoint>`
* `test_ndft.py -i <image> -o <omPath> -t <title> -a <adjoint>`
* `test_NFFT.py <file1> <file2> ...`

The first three files calls one library to compute the NFFT over the image (which path is given by `-i <image>` argument) with the (non-uniform) sampling given by `-o <omPath>.` argument. The results of forward (and/or adjoint) is stored in .npy files named after `-t ` argument.

* `test_pynufft.py` calls PyNufft module and compute forward and/or adjoint, in CPU or GPU mode (see below for options)
* `test_pynfft.py` calls Pynfft module and compute forward and/or adjoint. Beware, for pynfft to work, NFFT C library must be compiled on your machine
* `test_ndft.py` computes Direct non-uniform Fourier transform. Beware, as it is extrememly long, even if the code is optimized in Cython (with a x11 acceleration comparing to pure Python)

Datas must be stored in .npy format:
* images must be Ndim arrays normalized in [0,1] (3D volume by 256 = (256,256,256) numpy array, with np.float64 between 0 and 1)
* om (non uniform frequences samples): array of shape K*Ndim between [-0.5, 0.5[ for the sample locations.

Thoses files are stored in datas/ folder (must be created when the git project is cloned)

The others arguments are parameters of the transform (for exemple, size of interpolator, or subsampling grid. For more info, check documentation for thoses modules)

#### Calling a transform

The script will execute the NFFT on provided datas and will store result (k-space), and processing times in a Python dictionnary, which will be saved in a .npy file in the results/ folder.

* `-t <title>` is the title of the saved output file (useful to remember the configuration of each test)
* `-a <adjoint>` is a boolean, if True it will evaluate the adjoint transform on the k-space and save the reconstructed image in .npy file (plus processing times)
* `-g <gpu>` is a boolean, if True it will use not the CPU-only version of Pynufft, but  GPU-pynufft. Warning this one is still buggy.

#### Interpreting results
After generating files for each transform you want to compare, just call `test_NFFT.py` with the list of .npy files in argument. It will then plot the MSE and SSIM between computed k-space, and also plot processing times for each transform. The resulting figures are stored in folder results/ as .png image (folder results/ must be created when git project is cloned).

#### Scripted comparisons
The pipeline for some comparisons have been scripted in .sh files, such as:
* `compare_basic.sh`: Compute PyNufft and PyNFFT CPU transforms, adjoints with standard parameters (2D) and compare the results
* `compare_interp_pynufft.sh`: compare results of Pynufft transform aside different sizes of the interpolator kernel.
* `compare_gpu.sh`: compare performances of GPU Pynufft transform versus Pynfft
* `compare_cpu.sh`: compare performances between CPU and GPU Pynufft
* and others ...

Thoses scripts can be also seen as examples on how calling the Python scripts.

## III - Results and limitations
#### 2D transforms
With CPU functions, 2D transforms works well (SSIM is near 1 under a 1e-7 tolerance, which means the two modules computes near-identical transforms). However, PyNufft is significantly slower than PyNFFT, which isn't surprising for the CPU version, as it is mostly Python code versus a C kernel for PyNFFT.
![enter image description here](https://lh3.googleusercontent.com/CKnMrb93wIF-lMEYMQuSQmMnGqyzXxwOoYGijpnD5reVtnYMWZbn0xmngMaFNXpHLD2YtdMJ92BO "Performance comparison between CPU Nufft and NFFT on &#40;256,256&#41; image")
#### GPU Pynufft
GPU Pynufft, with a more complex syntax to call functions, is also performing not only poorly to PyNFFT but also CPU Pynufft.
As the graph below indicates, precomputation in GPU Pynufft is worse than in CPU Pynufft, and widely worse than PyNFFT. However, actual computation time is quicker.
![](https://lh3.googleusercontent.com/2Nw-K6ts4encFQSzBO9dK6YYfU1v1SxWRCAIob4RVSoihiz3f14kfmZV2sX12Mg0sxxAYmyb47P9 "Time performance of &#40;from left to right&#41; GPU PyNufft, CPU Pynufft, PyNFFT")

A closer look to the code highlights that it is the `plan()` function, computing specific matrices for later multiplications which is significantly slower.

#### 3D transforms

Another problem has been shown for transformations over 3D volumes. While it works well on the example furnished in the module, and on small volumes, the computation crashes if the volume size exceeds a specific limit ( $64^3$ seems to work, but $128^3$ crashes violently upon plan() computation; idem for GPU Pynufft)


## Conclusion
Contact author for debug help, especially over GPU calls
