#include "cuda_helper.h"
#include "ffthelper.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <math.h>
#include "cuComplex.h"

#include <complex>
#include <vector>

using ComplexVec = std::vector<std::complex<float>>;

namespace refft {

// Modular multiplication a * N mod p
// In: a[np][N]

// TWIDDLE CONSTANT
const int TWIDDLE_GRID_DIM = 256;
const int TWIDDLE_BLOCK_DIM = 128;

// TRANSPOSE CONSTANT
const int T_SMEM_SIZE = 8;
const int T_BLOCK_ROW = 4;
const int T_nx = 256;
const int T_ny = 128;

// FFT CONSTANT
const int FFT1_SIZE = 128;
const int FFT2_SIZE = 256;
const int BLOCK_DIM_1 = 32;
const int GRID_DIM_1 = 256;
const int BLOCK_DIM_2 = 64;
const int GRID_DIM_2 = 128;

__device__ cuFloatComplex twiddle(const float expr) {
  cuFloatComplex res;
  sincosf(expr, &res.y, &res.x);
  return res;
}

// RADIX-8 STOCKHAM ALGORITHM FFT
__device__ void fft_radix8(int n, cuFloatComplex *x, cuFloatComplex *constant) {
  const cuFloatComplex j = {0, 1};
  const cuFloatComplex c1 = {0.70711, 0.70711};
  const cuFloatComplex c2 = {0.70711, -0.70711};
  int s = 1;
  int m = n;

  const int n1 = n / 8;
  const int n2 = n / 4;
  const int n3 = n1 + n2;
  const int n4 = n / 2;
  const int n5 = n1 + n4;
  const int n6 = n3 + n3;
  const int n7 = n3 + n4;

  for(int i = 0; i < 3; i++) {
    int q = threadIdx.x % s;
    int p = (threadIdx.x - q) / s;

    cuFloatComplex w1 = constant[512 * p / m];
    cuFloatComplex w2 = cuCmulf(w1, w1);
    cuFloatComplex w3 = constant[512 * 3 * p / m];
    cuFloatComplex w4 = cuCmulf(w2, w2);
    cuFloatComplex w5 = constant[512 * 5 * p / m];
    cuFloatComplex w6 = cuCmulf(w3, w3);
    cuFloatComplex w7 = constant[517 * 7 * p / m];

    cuFloatComplex a = x[q + s * p];
    cuFloatComplex b = x[q + s * p + n1];
    cuFloatComplex c = x[q + s * p + n2];
    cuFloatComplex d = x[q + s * p + n3];
    cuFloatComplex e = x[q + s * p + n4];
    cuFloatComplex f = x[q + s * p + n5];
    cuFloatComplex g = x[q + s * p + n6];
    cuFloatComplex h = x[q + s * p + n7];

    cuFloatComplex ae0 = cuCaddf(a, e);
    cuFloatComplex ae1 = cuCsubf(a, e);
    cuFloatComplex cg0 = cuCaddf(c, g);
    cuFloatComplex cg1 = cuCmulf(j, cuCsubf(c, g));
    cuFloatComplex bf0 = cuCaddf(b, f);
    cuFloatComplex bf1 = cuCsubf(b, f);
    cuFloatComplex dh0 = cuCaddf(d, h);
    cuFloatComplex dh1 = cuCmulf(j, cuCsubf(d, h));

    cuFloatComplex aceg0 = cuCaddf(ae0, cg0);
    cuFloatComplex aceg1 = cuCsubf(ae1, cg1);
    cuFloatComplex aceg2 = cuCsubf(ae0, cg0);
    cuFloatComplex aceg3 = cuCaddf(ae1, cg1);
    cuFloatComplex bdfh0 = cuCaddf(bf0, dh0);
    cuFloatComplex bdfh1 = cuCmulf(c1, cuCsubf(bf1, dh1));
    cuFloatComplex bdfh2 = cuCmulf(j, cuCsubf(bf0, dh0));
    cuFloatComplex bdfh3 = cuCmulf(c2, cuCaddf(bf1, dh1));
    __syncthreads();

    x[q + s * 8 * p] = cuCaddf(aceg0, bdfh0);
    x[q + s * (8 * p + 1)] = cuCmulf(w1, cuCsubf(aceg1, bdfh1));
    x[q + s * (8 * p + 2)] = cuCmulf(w2, cuCsubf(aceg2, bdfh2));
    x[q + s * (8 * p + 3)] = cuCmulf(w3, cuCsubf(aceg3, bdfh3));
    x[q + s * (8 * p + 4)] = cuCmulf(w4, cuCsubf(aceg0, bdfh0));
    x[q + s * (8 * p + 5)] = cuCmulf(w5, cuCaddf(aceg1, bdfh1));
    x[q + s * (8 * p + 6)] = cuCmulf(w6, cuCaddf(aceg2, bdfh2));
    x[q + s * (8 * p + 7)] = cuCmulf(w7, cuCaddf(aceg3, bdfh3));
    __syncthreads();

    s = s << 3;
    m = m >> 3;
  }
}

// RADIX-4 STOCKHAM ALGORITHM FFT
__device__ void fft_radix4_even(int n, cuFloatComplex *x, cuFloatComplex *constant) {
  const cuFloatComplex j = {0, 1};
  int s = 1;
  int m = n;
  int k = 0;
  const int n1 = n/4;
  const int n2 = n/2;
  const int n3 = n1 + n2;

  for(int i = 0; i < 4; i++) {
    int q = threadIdx.x % s;
    int p = (threadIdx.x - q) >> k;

    cuFloatComplex w1 = constant[256 * p / m];
    cuFloatComplex w2 = cuCmulf(w1, w1);
    cuFloatComplex w3 = cuCmulf(w1, w2);
    __syncthreads();

    cuFloatComplex a = x[q + s * p];
    cuFloatComplex b = x[q + s * p + n1];
    cuFloatComplex c = x[q + s * p + n2];
    cuFloatComplex d = x[q + s * p + n3];
    cuFloatComplex temp1_even = cuCaddf(a, c);
    cuFloatComplex temp1_odd = cuCsubf(a, c);
    cuFloatComplex temp2_even = cuCaddf(b, d);
    cuFloatComplex temp2_odd = cuCmulf(j, cuCsubf(b, d));
    __syncthreads();
    
    x[q + s * 4 * p] = cuCaddf(temp1_even, temp2_even);
    x[q + s * (4 * p + 1)] = cuCmulf(w1, cuCsubf(temp1_odd, temp2_odd));
    x[q + s * (4 * p + 2)] = cuCmulf(w2, cuCsubf(temp1_even, temp2_even));
    x[q + s * (4 * p + 3)] = cuCmulf(w3, cuCaddf(temp1_odd, temp2_odd));
    __syncthreads();

    s = s << 2;
    m = m >> 2;
    k = k + 2;
  }
}

// RADIX-4 STOCKHAM ALGORITHM FFT
__device__ void fft_radix4_odd(int n, cuFloatComplex *x, cuFloatComplex *constant, cuFloatComplex *t) {
  const cuFloatComplex j = {0, 1};
  int s = 1;
  int m = n;
  int k = 0;
  const int n1 = n/4;
  const int n2 = n/2;
  const int n3 = n1 + n2;

  for(int i = 0; i < 3; i++) {
    int q = threadIdx.x % s;
    int p = (threadIdx.x - q) >> k;
    
    cuFloatComplex w1 = constant[256 * p / m];
    cuFloatComplex w2 = cuCmulf(w1, w1);
    cuFloatComplex w3 = cuCmulf(w1, w2);
    
    cuFloatComplex a = x[q + s * p];
    cuFloatComplex b = x[q + s * p + n1];
    cuFloatComplex c = x[q + s * p + n2];
    cuFloatComplex d = x[q + s * p + n3];
    cuFloatComplex temp1_even = cuCaddf(a, c);
    cuFloatComplex temp1_odd = cuCsubf(a, c);
    cuFloatComplex temp2_even = cuCaddf(b, d);
    cuFloatComplex temp2_odd = cuCmulf(j, cuCsubf(b, d));
    __syncthreads();

    x[q + s * 4 * p] = cuCaddf(temp1_even, temp2_even);
    x[q + s * (4 * p + 1)] = cuCmulf(w1, cuCsubf(temp1_odd, temp2_odd));
    x[q + s * (4 * p + 2)] = cuCmulf(w2, cuCsubf(temp1_even, temp2_even));
    x[q + s * (4 * p + 3)] = cuCmulf(w3, cuCaddf(temp1_odd, temp2_odd));
    __syncthreads();    

    s = s << 2;
    m = m >> 2;
    k = k + 2;
  }

  cuFloatComplex a = x[threadIdx.x];
  cuFloatComplex b = x[threadIdx.x + n2];
  cuFloatComplex c = x[threadIdx.x + n1];
  cuFloatComplex d = x[threadIdx.x + n3];
  cuFloatComplex temp1_even = cuCaddf(a, b);
  cuFloatComplex temp1_odd = cuCsubf(a, b);
  cuFloatComplex temp2_even = cuCaddf(c, d);
  cuFloatComplex temp2_odd = cuCsubf(c, d);
  __syncthreads();
  
  // twiddle
  cuFloatComplex w1 = t[threadIdx.x * blockIdx.x];
  cuFloatComplex w = t[BLOCK_DIM_1 * blockIdx.x];
  __syncthreads();

  cuFloatComplex w2 = cuCmulf(w1, w);//twiddle(theta * (threadIdx.x + BLOCK_DIM_1) * blockIdx.x);
  cuFloatComplex w3 = cuCmulf(w2, w);//twiddle(theta * (threadIdx.x + 2 * BLOCK_DIM_1) * blockIdx.x);
  cuFloatComplex w4 = cuCmulf(w, cuCmulf(w, w2));//twiddle(theta * (threadIdx.x + 3 * BLOCK_DIM_1) * blockIdx.x); 
  __syncthreads();  

  temp1_even = cuCmulf(w1, temp1_even);
  temp2_even = cuCmulf(w2, temp2_even);
  temp1_odd = cuCmulf(w3, temp1_odd);
  temp2_odd = cuCmulf(w4, temp2_odd);
  __syncthreads();

  x[threadIdx.x] =  temp1_even;
  x[threadIdx.x + n2] = temp1_odd;
  x[threadIdx.x + n1] = temp2_even;
  x[threadIdx.x + n3] = temp2_odd;
  __syncthreads();
}

__device__ void butt_fft(cuFloatComplex *a, cuFloatComplex *b,
                         cuFloatComplex w) {
  cuFloatComplex U = cuCmulf(*b, w);
  *b = cuCsubf(*a, U);
  *a = cuCaddf(*a, U);
}

__global__ void Cal(cuFloatComplex *a, const int N) {
  const float theta = - M_PI * (threadIdx.x + blockIdx.x * blockDim.x) / N;
  const cuFloatComplex c = twiddle(theta);
  a[threadIdx.x + blockIdx.x * blockDim.x] = c;
}

__global__ void Transpose(cuFloatComplex *a)
{
  __shared__ cuFloatComplex smem[T_SMEM_SIZE][T_SMEM_SIZE + 1];
    
  int x = blockIdx.x * T_SMEM_SIZE + threadIdx.x;
  int y = blockIdx.y * T_SMEM_SIZE + threadIdx.y;
  int width_x = gridDim.x * T_SMEM_SIZE;
  int width_y = gridDim.y * T_SMEM_SIZE;
  
  for (int i = 0; i < T_SMEM_SIZE; i += T_BLOCK_ROW)
    smem[threadIdx.y + i][threadIdx.x] = a[(y + i) * width_x + x];
  __syncthreads();

  x = blockIdx.y * T_SMEM_SIZE + threadIdx.x;
  y = blockIdx.x * T_SMEM_SIZE + threadIdx.y;

  for (int i = 0; i < T_SMEM_SIZE; i += T_BLOCK_ROW)
    a[(y + i) * width_y + x] = smem[threadIdx.x][threadIdx.y + i];
}

__global__ void Fft(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2);
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(-M_PI * (double)t_idx / (double)m);
    butt_fft(a_x, a_y, w);
  }
}

__global__ void FftWithTwiddle_Radix4(cuFloatComplex *a, cuFloatComplex *c, cuFloatComplex *t, const int N) {
  // shared memory
  __shared__ cuFloatComplex x[FFT1_SIZE];
  __shared__ cuFloatComplex constant[64];

  // global memory -> shared memory without shared memory bank conflict
  //x[threadIdx.x]                         = a[blockIdx.x * FFT1_SIZE + threadIdx.x];
  //x[threadIdx.x + BLOCK_DIM_1]           = a[blockIdx.x * FFT1_SIZE + BLOCK_DIM_1 + threadIdx.x];
  //x[threadIdx.x + 2 * BLOCK_DIM_1]       = a[blockIdx.x * FFT1_SIZE + 2 * BLOCK_DIM_1 + threadIdx.x];
  //x[threadIdx.x + 3 * BLOCK_DIM_1]       = a[blockIdx.x * FFT1_SIZE + 3 * BLOCK_DIM_1 + threadIdx.x];
  x[threadIdx.x]                           = a[blockIdx.x + FFT2_SIZE * (threadIdx.x)];
  x[threadIdx.x + BLOCK_DIM_1]             = a[blockIdx.x + FFT2_SIZE * (BLOCK_DIM_1 + threadIdx.x)];
  x[threadIdx.x + 2 * BLOCK_DIM_1]         = a[blockIdx.x + FFT2_SIZE * (2 * BLOCK_DIM_1 + threadIdx.x)];
  x[threadIdx.x + 3 * BLOCK_DIM_1]         = a[blockIdx.x + FFT2_SIZE * (3 * BLOCK_DIM_1 + threadIdx.x)];
  constant[threadIdx.x]                    = c[threadIdx.x];
  constant[threadIdx.x + blockDim.x]       = c[threadIdx.x + blockDim.x];
  __syncthreads();

  // FFT + Twiddle
  fft_radix4_odd(FFT1_SIZE, x, constant, t);

  // shared memory -> global memory without shared memory bank conflict
  //a[blockIdx.x * FFT1_SIZE + threadIdx.x]                   = x[threadIdx.x];
  //a[blockIdx.x * FFT1_SIZE + BLOCK_DIM_1 + threadIdx.x]     = x[threadIdx.x + BLOCK_DIM_1];
  //a[blockIdx.x * FFT1_SIZE + 2 * BLOCK_DIM_1 + threadIdx.x] = x[threadIdx.x + 2 * BLOCK_DIM_1];
  //a[blockIdx.x * FFT1_SIZE + 3 * BLOCK_DIM_1 + threadIdx.x] = x[threadIdx.x + 3 * BLOCK_DIM_1];
  a[blockIdx.x + FFT2_SIZE * (threadIdx.x)]                   = x[threadIdx.x];
  a[blockIdx.x + FFT2_SIZE * (BLOCK_DIM_1 + threadIdx.x)]     = x[threadIdx.x + BLOCK_DIM_1];
  a[blockIdx.x + FFT2_SIZE * (2 * BLOCK_DIM_1 + threadIdx.x)] = x[threadIdx.x + 2 * BLOCK_DIM_1];
  a[blockIdx.x + FFT2_SIZE * (3 * BLOCK_DIM_1 + threadIdx.x)] = x[threadIdx.x + 3 * BLOCK_DIM_1];
}

__global__ void FftWithoutTwiddle_Radix4(cuFloatComplex *a, cuFloatComplex *c) {
  // shared memory
  __shared__ cuFloatComplex x[FFT2_SIZE];
  __shared__ cuFloatComplex constant[64];

  // global memory -> shared memory without shared memory bank conflict
  x[threadIdx.x]                   = a[blockIdx.x * FFT2_SIZE + threadIdx.x];
  x[threadIdx.x + BLOCK_DIM_2]     = a[blockIdx.x * FFT2_SIZE + BLOCK_DIM_2 + threadIdx.x];
  x[threadIdx.x + 2 * BLOCK_DIM_2] = a[blockIdx.x * FFT2_SIZE + 2 * BLOCK_DIM_2 + threadIdx.x];
  x[threadIdx.x + 3 * BLOCK_DIM_2] = a[blockIdx.x * FFT2_SIZE + 3 * BLOCK_DIM_2 + threadIdx.x];
  constant[threadIdx.x]            = c[threadIdx.x];
  __syncthreads();

  // FFT
  fft_radix4_even(FFT2_SIZE, x, constant);

  // shared memory -> global memory without shared memory bank conflict
  a[blockIdx.x * FFT2_SIZE + threadIdx.x]                   = x[threadIdx.x];
  a[blockIdx.x * FFT2_SIZE + BLOCK_DIM_2 + threadIdx.x]     = x[threadIdx.x + BLOCK_DIM_2];
  a[blockIdx.x * FFT2_SIZE + 2 * BLOCK_DIM_2 + threadIdx.x] = x[threadIdx.x + 2 * BLOCK_DIM_2];
  a[blockIdx.x * FFT2_SIZE + 3 * BLOCK_DIM_2 + threadIdx.x] = x[threadIdx.x + 3 * BLOCK_DIM_2];
}

__global__ void FftStudent(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2) * 1;
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(-M_PI * (double)t_idx / (double)m);
    butt_fft(a_x, a_y, w);
  }
}

__device__ void butt_ifft(cuFloatComplex *a, cuFloatComplex *b,
                          cuFloatComplex w) {
  cuFloatComplex T = cuCsubf(*a, *b);
  *a = cuCaddf(*a, *b);
  (*a).x /= 2.0;
  (*a).y /= 2.0;
  *b = cuCmulf(T, w);
  (*b).x /= 2.0;
  (*b).y /= 2.0;
}

__global__ void Ifft(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2);
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(M_PI * (double)t_idx / (double)m);
    butt_ifft(a_x, a_y, w);
  }
}

__global__ void IfftStudent(cuFloatComplex *a, const int m, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N / 2);
       i += blockDim.x * gridDim.x) {
    // index in N/2 range
    int N_idx = i % (N / 2);
    // i'th block
    int m_idx = N_idx / m;
    // base address
    cuFloatComplex *a_np = a;
    int t_idx = N_idx % m;
    cuFloatComplex *a_x = a_np + 2 * m_idx * m + t_idx;
    cuFloatComplex *a_y = a_x + m;
    cuFloatComplex w = twiddle(M_PI * (double)t_idx / (double)m);
    butt_ifft(a_x, a_y, w);
  }
}

__global__ void bitReverse(std::complex<float> *a, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N);
       i += blockDim.x * gridDim.x) {
    int logN = __log2f(N);
    int N_idx = i % N;
    std::complex<float> *a_x = a;
    int revN = __brev(N_idx) >> (32 - logN);
    if (revN > N_idx) {
      std::complex<float> temp = a_x[N_idx];
      a_x[N_idx] = a_x[revN];
      a_x[revN] = temp;
    }
  }
}

__device__ cuFloatComplex Cmul(cuFloatComplex a, cuFloatComplex b) {
  float temp = double(a.x) * b.x - double(a.y) * b.y;
  float temp2 = double(a.x) * b.y + double(a.y) * b.x;
  cuFloatComplex res;
  res.x = temp;
  res.y = temp2;
  return res;
}

__global__ void Hadamard(cuFloatComplex *a, cuFloatComplex *b, int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N);
       i += blockDim.x * gridDim.x) {
    int N_idx = i % N;
    cuFloatComplex *a_x = a;
    cuFloatComplex *b_x = b;
    a_x[N_idx] = Cmul(a_x[N_idx], b_x[N_idx]);
  }
}

void FftHelper::ExecFft(std::complex<float> *a, int N) {
  dim3 blockDim(refft::FFTblocksize);
  dim3 gridDim(N/2/refft::FFTblocksize);
  bitReverse<<<gridDim, blockDim>>>(a,N);
  for (int i = 1; i < N; i *= 2) {
    Fft<<<gridDim, blockDim>>>((cuFloatComplex *)a, i, N);
    CudaCheckError();
  }
  CudaCheckError();
}

void FftHelper::ExecStudentFft(std::complex<float> *a, std::complex<float> *c, std::complex<float> *t, int N){
  dim3 blockDim1(BLOCK_DIM_1);
  dim3 gridDim1(GRID_DIM_1);
  dim3 blockDim2(BLOCK_DIM_2);
  dim3 gridDim2(GRID_DIM_2);
  dim3 gridDim3(T_nx/T_SMEM_SIZE, T_ny/T_SMEM_SIZE, 1);
  dim3 blockDim3(T_SMEM_SIZE, T_BLOCK_ROW, 1);
  
  Cal<<<64, 1>>>((cuFloatComplex *)c, 128);
  Cal<<<TWIDDLE_GRID_DIM, TWIDDLE_BLOCK_DIM>>>((cuFloatComplex *)t, N/2);
  FftWithTwiddle_Radix4<<<gridDim1, blockDim1>>>((cuFloatComplex *)a, (cuFloatComplex *)c, (cuFloatComplex *)t, N);
  FftWithoutTwiddle_Radix4<<<gridDim2, blockDim2>>>((cuFloatComplex *)a, (cuFloatComplex *)c);
  CudaCheckError();
  Transpose<<<gridDim3, blockDim3>>>((cuFloatComplex *)a);
  CudaCheckError();
}

void FftHelper::ExecIfft(std::complex<float> *a, int N) {
  dim3 blockDim(refft::iFFTblocksize);
  dim3 gridDim(N/2/refft::iFFTblocksize);
  for (int i = N / 2; i > 0; i >>= 1) {
    Ifft<<<gridDim, blockDim>>>((cuFloatComplex *)a, i, N);
  }
  bitReverse<<<gridDim, blockDim>>>(a, N);
  CudaCheckError();
}

void FftHelper::ExecStudentIfft(std::complex<float> *a, int N) {
  dim3 blockDim(refft::iFFTblocksize);
  dim3 gridDim(N/2/refft::iFFTblocksize);
  for (int i = N / 2; i > 0; i >>= 1) {
    IfftStudent<<<gridDim, blockDim>>>((cuFloatComplex *)a, i, N);
  }
  bitReverse<<<gridDim, blockDim>>>(a, N);
  CudaCheckError();
}

void FftHelper::Mult(std::complex<float> *a, std::complex<float> *b, int N) {
  dim3 blockDim(refft::iFFTblocksize);
  dim3 gridDim(N/refft::iFFTblocksize);
  Hadamard<<<gridDim, blockDim>>>((cuFloatComplex*)a,(cuFloatComplex*)b, N);  
  CudaCheckError();
}
}  // namespace refft
