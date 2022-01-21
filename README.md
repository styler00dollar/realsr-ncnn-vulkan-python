# RealSR ncnn Vulkan Python

This fork aims to have support for various models. Technically you can run any model, if your param file has the correct names. The current code looks like this and this should match your param file.
```
ex.input("data", in);
ex.extract("output", out);
```
This code was tested with the [compact](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.3.0) and [normal](https://github.com/nihui/realsr-ncnn-vulkan/tree/4cc88321f71c1b4731d84393c93740b551823779/models) models.

Install instructions
```bash
# dont use conda, CXX errors in manjaro otherwise
conda deactivate
git clone https://github.com/styler00dollar/realsr-ncnn-vulkan-python
cd realsr-ncnn-vulkan-python/realsr_ncnn_vulkan_python/realsr-ncnn-vulkan/
git submodule update --init --recursive
cd src

# There are 2 CMakeLists.txt
# Make sure that prelu is set to ON, otherwise the compact model wont work
    option(WITH_LAYER_prelu "" ON)



cmake -B build .
cd build
make -j8
sudo su
make install
exit
cd .. && cd .. && cd .. && cd ..
python setup.py install --user
```
You can also do some custom modifications
```python
# if you dont want the 2 default pth files in your whl / install,
# comment the lines with say "models" in CMakeLists.txt

# to force input -> output without any tiling, you can do this in realsr.cpp
    // insert image size
    const int TILE_SIZE_X = w;
    const int TILE_SIZE_Y = h;

    // forcing to process one tile
    const int xtiles = 1;
    const int ytiles = 1;
    
# if you dont want to use PIL, swap code in realsr_ncnn_vulkan_python
    #instead of 
    in_bytes = bytearray(im.tobytes())
    [...]
    Image.frombytes(
            im.mode,
            (self._raw_realsr.scale * im.width, self._raw_realsr.scale * im.height),
            bytes(out_bytes),
        )
    #you need to have
    import numpy as np
    import cv2
    in_bytes = bytearray(np.array(im).tobytes(order='C'))
    [...]
    out_numpy = np.frombuffer(bytes(out_bytes), dtype=np.uint8)
    out_numpy = np.reshape(out_numpy, (self._raw_realsr.scale * im.height, self._raw_realsr.scale * im.width, 3))
    
    # its in rgb, if you want to save with opencv, convert to bgr
    out_numpy = cv2.cvtColor(out_numpy, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_opencv.png", out_numpy)
```
Minimalistic example
```python
from PIL import Image
from tqdm import tqdm
from realsr_ncnn_vulkan_python import RealSR
from pathlib import Path
import time
import threading

param_path = "test.param"
bin_path = "test.bin"

generic_inference = RealSR(gpuid=0, scale=2, tta_mode=False, param_path=param_path, bin_path=bin_path)
image = Image.open("test.png")

for i in tqdm(range(1000)):
  output = generic_inference.process(image)
  output.save("output.png")
```

There can be overlapping execution problems. A simple fix is to run it in a thread.
```python
# demonstration of a hotfix to avoid overlapping execution
# depending on what code you compile, there seems to be overlapping, can be fixed by running in a thread
def f(image):
  image = generic_inference.process(image)
  image.save("output.png")

for i in tqdm(range(1000)):
  thread = threading.Thread(target=f, args=(image,))
  thread.start()
  thread.join()
  
# alternative
import concurrent.futures

def foo(image):
    return generic_inference.process(image)

for i in tqdm(range(1000)):
  with concurrent.futures.ThreadPoolExecutor() as executor:
      future = executor.submit(foo, image)
      output_image = future.result()
      output_image.save("output.png")
```

TODO:
- Remove needless code

## Introduction
[realsr-ncnn-vulkan](https://github.com/nihui/realsr-ncnn-vulkan) is nihui's ncnn implementation of Real-World Super-Resolution via Kernel Estimation and Noise Injection super resolution.

realsr-ncnn-vulkan-python wraps [realsr-ncnn-vulkan project](https://github.com/nihui/realsr-ncnn-vulkan) by SWIG to make it easier to integrate realsr-ncnn-vulkan with existing python projects.

## Downloads

Linux/Windos/Mac X86_64 build releases are available now.

However, for Linux distro with GLIBC < 2.29 (like Ubuntu 18.04), the ubuntu-1804 pre-built should be used.

## Build

First, you have to install python, python development package (Python native development libs in Visual Studio), vulkan SDK and SWIG on your platform. And then:

### Linux
```shell
git clone https://github.com/ArchieMeng/realsr-ncnn-vulkan-python.git
cd realsr-ncnn-vulkan-python
git submodule update --init --recursive
cmake -B build src
cd build
make
```

### Windows
I used Visual Studio 2019 and msvc v142 to build this project for Windows.

Install visual studio and open the project directory, and build. Job done.

The only problem on Windows is that, you cannot use [CMake for Windows](https://cmake.org/download/) to generate the Visual Studio solution file and build it. This will make the lib crash on loading.

The only way is [use Visual Studio to open the project as directory](https://www.microfocus.com/documentation/visual-cobol/vc50/VS2019/GUID-BE1C48AA-DB22-4F38-9644-E9B48658EF36.html), and build it from Visual Studio.

## About RealSR

Real-World Super-Resolution via Kernel Estimation and Noise Injection (CVPRW 2020)

https://github.com/jixiaozhong/RealSR

Xiaozhong Ji, Yun Cao, Ying Tai, Chengjie Wang, Jilin Li, and Feiyue Huang

*Tencent YouTu Lab*

Our solution is the **winner of CVPR NTIRE 2020 Challenge on Real-World Super-Resolution** in both tracks.

https://arxiv.org/abs/2005.01996

## Usages

### Example Program

```Python
from PIL import Image
from realsr_ncnn_vulkan import RealSR

im = Image.open("0.png")
upscaler = RealSR(0, scale=4)
out_im = upscaler.process(im)
out_im.save("temp.png")
```

If you encounter crash or error, try to upgrade your GPU driver

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Original RealSR NCNN Vulkan Project

- https://github.com/nihui/realsr-ncnn-vulkan

## Original RealSR Project

- https://github.com/jixiaozhong/RealSR

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
