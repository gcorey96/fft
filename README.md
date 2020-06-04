# CUDA FFT Reference code
This is the reference [Fast Fourier Transform(FFT)][fft_reference] code for computer architecture class 2020.\
Find "ExecFft" function in src/ffthelper.cu, where the radix-2 [Cooley-Tukey FFT][cooley-tukey] algorithm is implemented as a reference code. You are asked to change the function to optimize for the GPU architecture, using the [GPGPU-sim simulator][gpgpu-sim].

[class_page]: https://scale-snu.github.io/jekyll/update/2020/03/16/aca2020-lecture-01.html
[docker_image]: https://hub.docker.com/r/michael604/comparch_430.636_spring2020
[gpgpu-sim]: https://github.com/gpgpu-sim/gpgpu-sim_distribution
[fft_reference]: https://github.com/gajh-classes/FFT_reference
[cooley-tukey]: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
[machine_config]: https://github.com/gpgpu-sim/gpgpu-sim_distribution/tree/dev/configs/tested-cfgs/SM6_TITANX



## Overview
In this project, you are recommended to use the docker to set up the GPGPU-sim and execution environment. You will optimize your SW code or HW using the [GPGPU-sim simulator][gpgpu-sim].

The set-up for the project will be as follows.
1. Pull docker images
2. Build FFT_reference
3. set environmental variables(for [GPGPU_sim][gpgpu-sim])
4. Execute the FFT_reference application binary



## Install Docker & Pull Image
The following instructions are based on the Ubuntu 18.04 LTS. You may need *sudo* authority to build and run the docker container.

```
$ apt-get install docker.io
$ docker pull michael604/comparch_430.636_spring2020
$ docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name {NAME} michael604/comparch_430.636_spring2020
```
Necessary dependencies and properly built [GPGPU-sim][gpgpu-sim] (dev 4.0) is already included in the [docker image][docker_image].
You need to create the container with `-it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined` option if you want to use gdb inside the container. You may name your container with `--name {NAME}` option.
GPGPU-sim is built as a release option in the provided image, but you may recompile it according to your need.



## How to Build & Run
This part will explain the necessary procedures to properly build [FFT_reference][fft_reference] CUDA application and how to attach it to GPGPU-sim when executing the binary.\
We also provide automated script file `makestep.sh`, so you may also use the script.\
(Directory names in the following explanation is based on the [docker image][docker_image] that is provided.

### 1. Build
in `/root/FFT_reference` directory,
```bash
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64
$ make -j
```
Binary file named `fft_reference` will be generated in `/root/FFT_reference/build`

### 2. Run
You need to set the correct environment variable to properly attach the CUDA application to GPGPU-sim. You also have to locate the correct GPGPU-sim configuration files and matrix data.

##### Set Environmental Variable
You first need to set environmental variables correctly.\
In `/root/gpgpu-sim_distribution` directory,
```bash
$ source setup_environment {BUILD_TYPE}
```
The default `BUILD_TYPE` in the [docker image][docker_image] is release.

##### Locate Configuration Files & Matrix Data
You need to locate `config_{ARCH}_islip.cnt` & `gpgpusim.config` files from `/gpgpu-sim_distribution/configs/tested-cfgs`, to the directory that you are going to run the application.
Also, text files that contain the matrix data should also be located properly. The size of the matrix will be fixed as `32768`.

In `/root/FFT_reference` directory,
```bash
$ mkdir build/run && cd build/run
$ cp /root/gpgpu-sim_distrubution/configs/tested-cfgs/{SM_NUM}/* ./
$ cp /root/FFT_reference/test/*.txt ./
```
We will use [`SM6_TITANX`][machine_config] configuration as a default machine configuration for the [GPGPU-sim][gpgpu-sim]. We do not need to run power simulation, so you need to change the `power_simulation_enabled` value to 0 in `gpgpusim.config` file.

##### Run CUDA Application
Normally execute the application binary, and it will attach to GPGPU-sim properly.\
You can explicitly set the `blocksize` of the FFT for your function, by setting the input value when running the binary. The default value for FFT `blocksize` is 256. The `gridsize` of the FFT is calculated as `N/2/blocksize`. You may change the value of `gridsize` by modifying the code in `/root/src/ffthelper.cu`.
```bash
$ cd /root/FFT_reference/build/run
$ ./../fft_reference --fftb {FFT_blocksize}
```

At the end of the application execution, `gpgpu_simulation_total_cycle` is displayed as the following log. 
(This value is a simple addition to the code in the GPGPU-sim of the [docker image][docker_image], but does not exist in the original GPGPU-sim(dev 4.0) code. If you are not using the docker image provided, please refer to the last `gpu_tot_sim_cycle` in the execution log instead.)
```bash
...

gpgpu_simulation_total_cycle = xxxxxx cycles
GPGPU-Sim: synchronize waiting for inactive GPU simulation
GPGPU-Sim API: Stream Manager State
GPGPU-Sim: detected inactive GPU simulation thread
GPGPU-Sim: *** exit detected ***
```
The performance of the optimized code will be evaluated by the `gpgpu_simulation_total_cycle` value. 
The code will be evaluated with the cycles displayed above.

##### makestep.sh
in `/root/FFT_reference`, automated script `makestep.sh` is provided. You may use the script to skip the above-explained procedures.



## Implementation & Evaluation
You can take either of the following two directions in conducting
this term project:

1. SW only approach: Optimize the **FFT** code with higher performance in the given `SM6_TITANX` Pascal configuration. The matrix size that you have to optimize your code with, is fixed as `32768`. Else than the configuration values, you can modify every part of the code, and add any additional custom functions. However, it is necessary to explain your implementation in your report.

2. HW modification: Modify the machine description file(`SM6_TITANX`) or GPGPU-sim simulator to suggest and implement a better hardware.

* When you run the application binary with GPGPU-sim, `gpgpu_simulation_total_cycle` is displayed at the end of the log. The performance of your optimized code and application will be evaluated accordingly.

* If it's necessary, you are allowed to modify the CMakeLists.txt. However, do not modify any important flags, such as optimization level flags. You also have to notify and explain what you have changed.



## Report

1. SW only approach: Describe your method of code optimization.
```
 => total_cycle = 26550 cycle

 Alogorithm
 
 0. Twiddle Factor 계산 (4931 cycle)

 1. Transpose (2398 cycle)
 
 2. 256개의 row에 대한 128개 coefficient FFT (radix-4 stockham algorithm) + twiddle factor 곱하기 (9393 cycle)

 3. Transpose (2164 cycle)

 4. 128개의 row에 대한 256개 coefficient FFT (radix-4 stockham algorithm) (5866 cycle)

 5. Transpose (1798 cycle)
 
  이번 과제에서는 Coefficient들에 대한 row major access만 가능한 것으로 간주하여 1, 5 단계의 transpose가 필요하지만
 column major access로 Coefficient들을 주고받을 수 있다면 1, 5단계의 transpose도 생략이 가능할 것으로 보입니다.

   0번째 단계에서 말하는 Twiddle Facotr 계산에는 2가지 종류가 있습니다.
 
  A. 128개와 256개 Coefficient에 대한 FFT를 진행할 때에 필요한 complex exponential 계산 (2846 cycle)
  
  B. 32768개의 Twiddle Factor 계산 (2085 cycle)

   A는 총 64개의 complex exponential만 계산하면 되는 단계로 B와 비교해서 필요한 계산량이 훨씬 작지만 더 많은 
  cycle을 필요로 하고 있습니다. 계산량이 너무 적어 bandwidth를 비효율적으로 쓰는 등의 문제가 있는 것으로 생각됩니다.
   따라서 A는 CPU에서 계산하고 이를 CUDA로 옮겨오는 것이 더 바람직한 구현이겠지만 이번 과제에서는 CPU에서의 연산이
  total cycle에 반영되지 않기 때문에 이렇게 구현할 수 없었습니다.
  
   0번재 단계에서 계산한 complex exponential 값들은 각각 shared memomry와 texture memory에 올라가 2, 4 단계가
  진행될 때 가져와서 사용하게 됩니다.

   32768개의 Coefficient FFT를 왜 128개 Coefficient들과 256개 Coefficient들에 대한 2차원 FFT로 radix-4 stockham
  algorithm을 이용하여 진행하였는지에 대해서는 곱해서 32768이 되는 2의 거듭제곱 쌍들은 ..., (32, 1024), (64, 512),
  (128, 256), (256, 128), ... 으로 많은 경우의 수가 존재하고 radix도 radix-2, radix-4, radix-8, ... 등이 존재합니다.
  저는 radix-2, radix-4, radix-8를 적용하여 주어진 FFT를 각각의 2의 거듭제곱 쌍들로 이루어진 2차원 FFT를 수행하여
  보았고, 그 중에서 total_cycle이 작은 것을 선택하였습니다.

   또한, Cooley-Tuckey Algorithm이 아니라 Stockham Algorithm을 쓴 이유는 Stockham Algorithm은 bit reverse를 따로
  수행하지 않아 성능이 더 잘 나오는 것을 확인하였기 때문입니다. 이러한 사실은 Reference 논문들에서 찾아볼 수 있습니다. 
  
   만약, Coefficient가 32768개가 아니라 더 많거나 더 적다고 한다면 어떻게 2차원으로 쪼개서 계산해야 되는지에 대 
  한 방법론은 알 수 없겠지만 (Reference에 있는 Auto-tunning of Fast Fourier Transform on Graphics Processors을
  참고한다면 어떻게 선택을 해야 할지에 대한 감 정도는 얻을 수 있을 것으로 생각됩니다.) 적어도 Coefficient의 개수가
  2의 거듭제곱 꼴로 바뀌었을 때 최대한 Reconfigurable하도록 CUDA Code를 짜려고 노력하였고 GridDimd이나 BlockDim과
  같은 주요한 변수들은 Parameterize하여 사용하였습니다.

   그 외에도 여러 가지 자잘한 Optimization을 하였는데 이 부분은 CUDA Code를 보시면 확인하실 수 있을 것으로 생각됩니다.

   Reference
   
   1. A GPU Based Memory Optimized Parallel Method For FFT Implementation
   2. Auto-tunning of Fast Fourier Transform on Graphics Processors
   3. Bandwidth Intensive 3-D FFT kernel for GPUs using CUDA
   4. Fast Computation of General Fourier Transforms on GPUs
   5. FFT algorithms for vector computers
   6. FFTs in external or hierarchical memory
   7. GPU Fast Convolution via the Overlap-and-Save Method in Shared Memory
   8. High performance discrete Fourier transforms on graphics processors
   9. Memory Locality Exploitation Strategies for FFt on the CUDA Architecture
```
2. HW modification: Describe your suggested hardware, and the reasoning behind the modifications.

* If you have not used the provided [docker image][docker_image], please specify the environment that you have worked in.

## Contact

If you find any errors or issues in the project, feel free to e-mail us. Our e-mails are as below, and are also posted on our [class description page][class_page].

Michael Jaemin Kim: michael604@scale.snu.ac.kr\
Sangpyo Kim: spkim@scale.snu.ac.kr\
