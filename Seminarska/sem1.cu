#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define err(msg, reason) { fprintf(stderr, "Error: %s %s\n", msg, reason); exit(1); }
#define COLOR_CHANNELS 1
#define GRAY_LEVELS 256
#define BLOCK_SIZE 1024

// calculate histogram
__global__ void calculateHistogram(unsigned char *image, int width, int height, unsigned int *histogram) {

    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x,
        localY = threadIdx.y;
    
    __shared__ unsigned int localHisto[GRAY_LEVELS];

    localHisto[blockDim.x * localY + localX] = 0;
    __syncthreads();

    if (x < width && y < height) {
        atomicAdd(&localHisto[image[y * width + x]], 1);
    }
    __syncthreads();

    atomicAdd(&(histogram[blockDim.x * localY + localX]), localHisto[blockDim.x * localY + localX]);

}

__global__ void parallelPrefixSum(unsigned int *histogram, unsigned int *cdf) {

    __shared__ unsigned int tmp[GRAY_LEVELS * 2];
    int threadID = threadIdx.x,
        pout = 0,
        pin = 1;
    
    tmp[threadID] = histogram[threadID];
    __syncthreads();

    for (int offset = 1; offset<GRAY_LEVELS; offset<<=1) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (threadID >= offset) {
            tmp[pout * GRAY_LEVELS + threadID] = tmp[pin * GRAY_LEVELS + threadID] + tmp[pin * GRAY_LEVELS + threadID - offset];
        } else {
            tmp[pout * GRAY_LEVELS + threadID] = tmp[pin * GRAY_LEVELS + threadID];
        }
        __syncthreads();
    }

    cdf[threadID] = tmp[pout * GRAY_LEVELS + threadID];

}

// make use of shared memory
__global__ void calcuateMinReduction(unsigned int *cdf, unsigned int *minimum) {

    // extern __shared__ unsigned int sharedData[];
    int threadID = threadIdx.x;

    // x >> 1 is the same as x / 2
    // and becouse bitwise op. are faster
    // we are gonna use it.
    // So it is stride / 2; stride>0; stride /= 2
    // this reduces the number of threads involved in the reduction
    // by half each iteration until there is only one left that performs
    // the final combination
    for (unsigned int stride = blockDim.x >> 1; stride>0; stride >>= 1) {
        // this is reduction
        __syncthreads();
        if (threadID < stride) {
            cdf[threadID] = (cdf[threadID] != 0) ?  min(cdf[threadID], cdf[threadID + stride]) : cdf[threadID + stride];
        }
    }
    // one thread remaining
    if (threadID == 0) {
        minimum[blockIdx.x] = cdf[0];
    }

}

__global__ void findMinGPU(unsigned int *cdf, unsigned int *min)
{
    int threadNumber = blockDim.x;
    __shared__ unsigned long tmp[GRAY_LEVELS];

    tmp[threadIdx.x] = cdf[threadIdx.x];

    __syncthreads();

    int i = threadNumber / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            if (tmp[threadIdx.x] < tmp[threadIdx.x + i])
            {
                if (tmp[threadIdx.x] > 0)
                {
                    tmp[threadIdx.x] = tmp[threadIdx.x];
                }
            }
            else
            {
                if (tmp[threadIdx.x + i] > 0)
                {
                    tmp[threadIdx.x] = tmp[threadIdx.x + i];
                }
            }
        }
        i = i / 2;
        __syncthreads();
    }
}

__device__ inline unsigned char scale(unsigned int cdf, unsigned int min, unsigned int imageSize) {

    float scale;

    scale = (float)(cdf - min) / (float)(imageSize - min);

    scale = round(scale * (float)(GRAY_LEVELS - 1));

    return (int) scale;

}

__global__ void equalize(unsigned char *imageIn, unsigned char *imageOut, int width, int height, unsigned int *cdf, unsigned int *cdfMin) {

    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    int localX = threadIdx.x,
        localY = threadIdx.y;

    unsigned int imageSize = width * height;

    __shared__ unsigned int localCdf[GRAY_LEVELS];

    localCdf[blockDim.x * localY + localX] = cdf[blockDim.x * localY + localX];
    __syncthreads();

    // unsigned int cdfMin = calcuateMinReduction(localCdf);

    if (x < width && y < height) {
        imageOut[y * width + x] = scale(localCdf[imageIn[y * width + x]], *cdfMin, imageSize); // cdf, cdfMin, imageSize
    }

}


int main(int argc, char **argv){

    // check for arguments
    if (argc < 2) {
        err("To few arguments.", "Usage: <image_in> <image_out>")
    }

    // host variables
    int width, height, imageSizeCuda, cpp = 1;
    unsigned char *imageIn, *imageOut;
    unsigned int *histogram, *cdf, *minimum;
    size_t imageSize;

    // device variables
    unsigned char *imageInDevice, *imageOutDevice;
    unsigned int *histogramDevice, *cdfDevice, *minimumDevice;

    // load image from file
    if (!(imageIn = stbi_load(argv[1], &width, &height, &cpp, 1))) {
        err("stbi_load", "Could not load image")
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", argv[1], width, height, cpp);

    // allocate memory for histogram
    if (!(histogram = (unsigned int *) calloc(GRAY_LEVELS, sizeof(unsigned int)))) { // SET IT TO ZERO
        err("calloc", "Could not allocate memory for histogram on host")
    }

    // allocate memory for cdf
    if (!(cdf = (unsigned int *) calloc(GRAY_LEVELS, sizeof(unsigned int)))) { // SET IT TO 0
        err("calloc", "Could not allocate memory for cdf on host")
    }
    /*

    /*
    if (!(newCdf = (unsigned int *) calloc(GRAY_LEVELS, sizeof(unsigned int)))) { // SET IT TO 0
        err("calloc", "Could not allocate memory for newCdf on host")
    }
    */

    minimum = (unsigned int *) calloc(1, sizeof(unsigned int));

    // calculate needed size of memory for image and allocate it
    imageSizeCuda = width * height * 1; // used on GPU as lenght of array
    imageSize = width * height * 1 * sizeof(unsigned int);
    if (!(imageOut = (unsigned char *) malloc(imageSize))) {
        err("malloc", "Could not allocate memory for output image on host")
    }

    // allocate memory on GPU
    // image
    cudaMalloc(&imageInDevice, imageSize);
    // histogram
    cudaMalloc(&histogramDevice, GRAY_LEVELS * sizeof(unsigned int));
    cudaMemset(histogramDevice, 0, GRAY_LEVELS * sizeof(unsigned int));
    // cdf
    cudaMalloc(&cdfDevice, GRAY_LEVELS * sizeof(unsigned int));
    cudaMemset(cdfDevice, 0, GRAY_LEVELS * sizeof(unsigned int));
    // minimum
    cudaMalloc(&minimumDevice, 1 * sizeof(unsigned int));
    // cudaMemset(minimumDevice, 0, 1 * sizeof(unsigned int));
    // imageOut
    cudaMalloc(&imageOutDevice, imageSize);
    cudaMemset(imageOutDevice, 0, imageSize);

    // copy data to device
    // image
    cudaMemcpy(imageInDevice, imageIn, imageSize, cudaMemcpyHostToDevice);
    /*
    // histogram
    cudaMemcpy(histogramDevice, histogram, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cdf
    cudaMemcpy(cdfDevice, cdf, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    */

    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / blockSize.x), ceil(height / blockSize.y));
    

    // int gridDim = (imageSizeCuda + 1) / BLOCK_SIZE;

    // run calculateHistogram
    calculateHistogram<<<gridSize,blockSize>>>(imageInDevice, width, height, histogramDevice);
    
    parallelPrefixSum<<<1,GRAY_LEVELS>>>(histogramDevice, cdfDevice);

    calcuateMinReduction<<<1,GRAY_LEVELS>>>(cdfDevice, minimumDevice);
    
    equalize<<<gridSize,blockSize>>>(imageInDevice, imageOutDevice, width, height, cdfDevice, minimumDevice);

    // run parallerPrexifSum
    // run calcuateMinReduction
    // calcuateMinReduction<<<gridSize,blockSize>>>(cdfDevice, );
    // run equalize

    // copy result from device to host
    // histogram
    cudaMemcpy(histogram, histogramDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(minimum, minimumDevice, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cdf, cdfDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(imageOut, imageOutDevice, imageSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i<GRAY_LEVELS; i++) {
        printf("Color %d has %ld pixels\n", i, histogram[i]);
    }

    printf("\n");
    for (int i = 0; i<GRAY_LEVELS; i++) {
        printf("How many px have color %d? %ld\n", i, cdf[i]);
    }

    printf("\n");
    printf("\nMIN IS: %ld\n\n", *minimum);

    /*
    cudaMemcpy(cdfDevice, cdf, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(originalCdfDevice, cdf, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(minimumDevice, minimum, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    calcuateMinReduction<<<gridSize,blockSize>>>(cdfDevice, minimumDevice);

    */

    // cdf
    // cudaMemcpy(cdf, cdfDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /*
    cudaMemcpy(minimum, minimumDevice, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    scale<<<1,GRAY_LEVELS>>>(originalCdfDevice, newCdfDevice, minimumDevice, imageSizeCuda);

    cudaMemcpy(newCdf, newCdfDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    */

    /*
    int sumOfPixles = 0;
    printf("GRAYLEVELS: %d\n", GRAY_LEVELS);
    for (int i = 0; i<GRAY_LEVELS; i++) {
        sumOfPixles += histogram[i];
        printf("Color %d has %ld pixels\n", i, histogram[i]);
    }
    printf("Sum of pixels: %d\nHEIGHT x WIDTH %d\nSAME: %d\n\n",
            sumOfPixles, height * width, sumOfPixles == height * width);

    printf("CDF:\n");
    for (int i = 0; i<GRAY_LEVELS; i++) {
        printf("How many px have color %d? %ld\n", i, cdf[i]);
    }

    printf("\nMIN IS: %ld\n\n", *minimum);
    
    printf("%ld\n", newCdf[0]);
    */
    // write image (final step)
    stbi_write_jpg(argv[2], width, height, COLOR_CHANNELS, imageOut, 100);

    // free allocated memory
    free(imageIn);
    free(imageOut);
    free(histogram);
    free(cdf);
    free(minimum);

    cudaFree(imageInDevice);
    cudaFree(imageOutDevice);
    cudaFree(histogramDevice);
    cudaFree(cdfDevice);
    cudaFree(minimumDevice);

    return 0;
}