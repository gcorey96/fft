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

// TRANSPOSE CONSTANT
const int T_SMEM_SIZE_1 = 16;
const int T_BLOCK_ROW_1 = 16;
const int T_SMEM_SIZE_2 = 16;
const int T_BLOCK_ROW_2 = 16;
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

__device__ void twd(const int N, cuFloatComplex *x) {
  const double theta = -2 * M_PI/N;
  cuFloatComplex w1 = twiddle(theta * threadIdx.x * blockIdx.x);
  cuFloatComplex w2 = twiddle(theta * (threadIdx.x + BLOCK_DIM_1) * blockIdx.x);
  cuFloatComplex w3 = twiddle(theta * (threadIdx.x + 2 * BLOCK_DIM_1) * blockIdx.x);
  cuFloatComplex w4 = twiddle(theta * (threadIdx.x + 3 * BLOCK_DIM_1) * blockIdx.x); 
  
  cuFloatComplex a = cuCmulf(w1, x[threadIdx.x]);
  cuFloatComplex b = cuCmulf(w2, x[threadIdx.x + BLOCK_DIM_1]);
  cuFloatComplex c = cuCmulf(w3, x[threadIdx.x + 2 * BLOCK_DIM_1]);
  cuFloatComplex d = cuCmulf(w4, x[threadIdx.x + 3 * BLOCK_DIM_1]);
  __syncthreads();

  x[threadIdx.x] = a;
  x[threadIdx.x + BLOCK_DIM_1] = b;
  x[threadIdx.x + 2 * BLOCK_DIM_1] = c;
  x[threadIdx.x + 3 * BLOCK_DIM_1] = d;
  __syncthreads();
}

// RADIX-4 STOCKHAM ALGORITHM FFT
__device__ void fft1_radix4(int n, cuFloatComplex *x) {
  cuFloatComplex j; j.x = 0; j.y = 1;
  int s = 1;
  int m = n;
  int k = 0;
  const int n1 = n/4;
  const int n2 = n/2;
  const int n3 = n1 + n2;

  for(int i = 0; i < 4; i++) {
    double theta = 2 * M_PI / m;
    int q = threadIdx.x % s;
    int p = (threadIdx.x - q) >> k;

    cuFloatComplex w1 = twiddle(- p * theta);
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
}

// RADIX-4 STOCKHAM ALGORITHM FFT
__device__ void fft2_radix4(int n, cuFloatComplex *x) {
  cuFloatComplex j; j.x = 0; j.y = 1;
  int s = 1;
  int m = n;
  int k = 0;
  const int n1 = n/4;
  const int n2 = n/2;
  const int n3 = n1 + n2;

  for(int i = 0; i < 3; i++) {
    double theta = 2*M_PI/m;
    int q = threadIdx.x % s;
    int p = (threadIdx.x - q) >> k;
    
    cuFloatComplex w1 = twiddle(- p * theta);
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
  cuFloatComplex b = x[threadIdx.x + (n >> 1)];
  cuFloatComplex c = x[threadIdx.x + (n >> 2)];
  cuFloatComplex d = x[threadIdx.x + (n >> 1) + (n >> 2)];
  cuFloatComplex temp1_even = cuCaddf(a, b);
  cuFloatComplex temp1_odd = cuCsubf(a, b);
  cuFloatComplex temp2_even = cuCaddf(c, d);
  cuFloatComplex temp2_odd = cuCsubf(c, d);
  __syncthreads();

  x[threadIdx.x] = temp1_even;
  x[threadIdx.x + (n >> 1)] = temp1_odd;
  x[threadIdx.x + (n >> 2)] = temp2_even;
  x[threadIdx.x + (n >> 1) + (n >> 2)] = temp2_odd;
  __syncthreads();
}

__device__ void butt_fft(cuFloatComplex *a, cuFloatComplex *b,
                         cuFloatComplex w) {
  cuFloatComplex U = cuCmulf(*b, w);
  *b = cuCsubf(*a, U);
  *a = cuCaddf(*a, U);
}

__global__ void Transpose1(cuFloatComplex *a)
{
  __shared__ cuFloatComplex smem[T_SMEM_SIZE_1][T_SMEM_SIZE_1 + 1];
    
  int x = blockIdx.x * T_SMEM_SIZE_1 + threadIdx.x;
  int y = blockIdx.y * T_SMEM_SIZE_1 + threadIdx.y;
  int width_x = gridDim.x * T_SMEM_SIZE_1;
  int width_y = gridDim.y * T_SMEM_SIZE_1;
  
  for (int i = 0; i < T_SMEM_SIZE_1; i += T_BLOCK_ROW_1)
    smem[threadIdx.y + i][threadIdx.x] = a[(y + i) * width_x + x];

  __syncthreads();

  x = blockIdx.y * T_SMEM_SIZE_1 + threadIdx.x;
  y = blockIdx.x * T_SMEM_SIZE_1 + threadIdx.y;

  for (int i = 0; i < T_SMEM_SIZE_1; i += T_BLOCK_ROW_1)
    a[(y + i) * width_y + x] = smem[threadIdx.x][threadIdx.y + i];
}

__global__ void Transpose2(cuFloatComplex *a)
{
  __shared__ cuFloatComplex smem[T_SMEM_SIZE_2][T_SMEM_SIZE_2 + 1];

  int x = blockIdx.x * T_SMEM_SIZE_2 + threadIdx.x;
  int y = blockIdx.y * T_SMEM_SIZE_2 + threadIdx.y;
  int width_x = gridDim.x * T_SMEM_SIZE_2;
  int width_y = gridDim.y * T_SMEM_SIZE_2;

  for (int i = 0; i < T_SMEM_SIZE_2; i += T_BLOCK_ROW_2)
    smem[threadIdx.y + i][threadIdx.x] = a[(y + i) * width_x + x];

   __syncthreads();
 
   x = blockIdx.y * T_SMEM_SIZE_2 + threadIdx.x;
   y = blockIdx.x * T_SMEM_SIZE_2 + threadIdx.y;
 
   for (int i = 0; i < T_SMEM_SIZE_2; i += T_BLOCK_ROW_2)
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

__global__ void FftWithTwiddle_Radix4(cuFloatComplex *a, const int N) {
    //for(int i = 0; i < 2; i++) {
      // shared memory
      extern __shared__ cuFloatComplex x[];
    
      // global memory -> shared memory without shared memory bank conflict
      x[threadIdx.x]                   = a[blockIdx.x * FFT1_SIZE + threadIdx.x];
      x[threadIdx.x + BLOCK_DIM_1]     = a[blockIdx.x * FFT1_SIZE + BLOCK_DIM_1 + threadIdx.x];
      x[threadIdx.x + 2 * BLOCK_DIM_1] = a[blockIdx.x * FFT1_SIZE + 2 * BLOCK_DIM_1 + threadIdx.x];
      x[threadIdx.x + 3 * BLOCK_DIM_1] = a[blockIdx.x * FFT1_SIZE + 3 * BLOCK_DIM_1 + threadIdx.x];
      __syncthreads();       

      // FFT + Twiddle
      fft2_radix4(FFT1_SIZE, x);
      twd(N, x);

      // shared memory -> global memory without shared memory bank conflict
      a[blockIdx.x * FFT1_SIZE + threadIdx.x]                   = x[threadIdx.x];
      a[blockIdx.x * FFT1_SIZE + BLOCK_DIM_1 + threadIdx.x]     = x[threadIdx.x + BLOCK_DIM_1];
      a[blockIdx.x * FFT1_SIZE + 2 * BLOCK_DIM_1 + threadIdx.x] = x[threadIdx.x + 2 * BLOCK_DIM_1];
      a[blockIdx.x * FFT1_SIZE + 3 * BLOCK_DIM_1 + threadIdx.x] = x[threadIdx.x + 3 * BLOCK_DIM_1];
    //}
}

__global__ void FftWithoutTwiddle_Radix4(cuFloatComplex *a) {
  //for(int i = 0; i < 2; i++) {  
    // shared memory
    extern __shared__ cuFloatComplex x[];

    // global memory -> shared memory without shared memory bank conflict
    x[threadIdx.x]                   = a[blockIdx.x * FFT2_SIZE + threadIdx.x];
    x[threadIdx.x + BLOCK_DIM_2]     = a[blockIdx.x * FFT2_SIZE + BLOCK_DIM_2 + threadIdx.x];
    x[threadIdx.x + 2 * BLOCK_DIM_2] = a[blockIdx.x * FFT2_SIZE + 2 * BLOCK_DIM_2 + threadIdx.x];
    x[threadIdx.x + 3 * BLOCK_DIM_2] = a[blockIdx.x * FFT2_SIZE + 3 * BLOCK_DIM_2 + threadIdx.x];
    __syncthreads();

    // FFT
    fft1_radix4(FFT2_SIZE, x);

    // shared memory -> global memory without shared memory bank conflict
    a[blockIdx.x * FFT2_SIZE + threadIdx.x]                   = x[threadIdx.x];
    a[blockIdx.x * FFT2_SIZE + BLOCK_DIM_2 + threadIdx.x]     = x[threadIdx.x + BLOCK_DIM_2];
    a[blockIdx.x * FFT2_SIZE + 2 * BLOCK_DIM_2 + threadIdx.x] = x[threadIdx.x + 2 * BLOCK_DIM_2];
    a[blockIdx.x * FFT2_SIZE + 3 * BLOCK_DIM_2 + threadIdx.x] = x[threadIdx.x + 3 * BLOCK_DIM_2];
  //}
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

/*
void FftHelper::ExecStudentFft(std::complex<float> *a, int N) {
  //dim3 blockDim(refft::FFTblocksize);
  //dim3 gridDim(N/2/refft::FFTblocksize);
  dim3 blockDim1(256);
  dim3 gridDim1(64);

  bitReverse<<<gridDim1, blockDim1>>>(a,N);
  for (int i = 1; i < N; i *= 2) {
    FftStudent<<<gridDim1, blockDim1>>>((cuFloatComplex *)a, i, N);
    CudaCheckError();
  }
  CudaCheckError();
}
*/

void FftHelper::ExecStudentFft(std::complex<float> *a, int N){
  dim3 gridDim1(T_nx/T_SMEM_SIZE_1, T_ny/T_SMEM_SIZE_1, 1);
  dim3 blockDim1(T_SMEM_SIZE_1, T_BLOCK_ROW_1, 1);
  dim3 blockDim2(BLOCK_DIM_1);
  dim3 gridDim2(GRID_DIM_1);
  dim3 gridDim3(T_ny/T_SMEM_SIZE_2, T_nx/T_SMEM_SIZE_2, 1);
  dim3 blockDim3(T_SMEM_SIZE_2, T_BLOCK_ROW_2, 1);
  dim3 blockDim4(BLOCK_DIM_2);
  dim3 gridDim4(GRID_DIM_2);
  dim3 gridDim5(T_nx/T_SMEM_SIZE_1, T_ny/T_SMEM_SIZE_1, 1);
  dim3 blockDim5(T_SMEM_SIZE_1, T_BLOCK_ROW_1, 1);

  Transpose1<<<gridDim1, blockDim1>>>((cuFloatComplex *)a);
  FftWithTwiddle_Radix4<<<gridDim2, blockDim2, 4 * BLOCK_DIM_1 * sizeof(cuFloatComplex)>>>((cuFloatComplex *)a, N);
  CudaCheckError();
  Transpose2<<<gridDim3, blockDim3>>>((cuFloatComplex *)a);
  FftWithoutTwiddle_Radix4<<<gridDim4, blockDim4, 4 * BLOCK_DIM_2 * sizeof(cuFloatComplex)>>>((cuFloatComplex *)a);
  CudaCheckError();
  Transpose1<<<gridDim5, blockDim5>>>((cuFloatComplex *)a);
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
