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
__global__ void calculateHistogram(unsigned char *image, int imageSize, unsigned int *histogram) {

    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    while (threadID < imageSize) {
        atomicAdd(&(histogram[image[threadID]]), 1);
        threadID += stride;
    }


}

/*
__global__ void parallelPrefixSum() {}
*/

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

__global__ void equalize() {}

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
    unsigned char *imageInDevice;
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

    minimum = (unsigned int *) calloc(1, sizeof(unsigned int));

    // calculate needed size of memory for image and allocate it
    imageSizeCuda = width * height * 1; // used on GPU as lenght of array
    imageSize = width * height * 1 * sizeof(unsigned long);
    if (!(imageOut = (unsigned char *) malloc(imageSize))) {
        err("malloc", "Could not allocate memory for output image on host")
    }

    // allocate memory on GPU
    // image
    cudaMalloc(&imageInDevice, imageSize);
    // histogram
    cudaMalloc(&histogramDevice, GRAY_LEVELS * sizeof(unsigned int));
    // SET IT TO ZERO!
    cudaMemset(histogramDevice, 0, GRAY_LEVELS * sizeof(unsigned int));
    // cdf
    cudaMalloc(&cdfDevice, GRAY_LEVELS * sizeof(unsigned int));
    // SET IT TO ZERO!
    // cudaMemset(cdfDevice, 0, GRAY_LEVELS * sizeof(unsigned int));
    // minimum
    cudaMalloc(&minimumDevice, 1 * sizeof(unsigned int));
    // cudaMemset(minimumDevice, 0, 1 * sizeof(unsigned int));

    // copy data to device
    // image
    cudaMemcpy(imageInDevice, imageIn, imageSize, cudaMemcpyHostToDevice);
    // histogram
    cudaMemcpy(histogramDevice, histogram, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cdf
    // cudaMemcpy(cdfDevice, cdf, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 blockSize(256, 1, 1);
    dim3 gridSize(1024 / 256, 1, 1);
    

    // int gridDim = (imageSizeCuda + 1) / BLOCK_SIZE;

    // run calculateHistogram
    calculateHistogram<<<gridSize,blockSize>>>(imageInDevice, imageSizeCuda, histogramDevice);
    // run parallerPrexifSum
    // run calcuateMinReduction
    // calcuateMinReduction<<<gridSize,blockSize>>>(cdfDevice, );
    // run equalize

    // copy result from device to host
    // histogram
    cudaMemcpy(histogram, histogramDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    for (int i=1; i<GRAY_LEVELS; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }

    cudaMemcpy(cdfDevice, cdf, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(minimumDevice, minimum, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    calcuateMinReduction<<<gridSize,blockSize>>>(cdfDevice, minimumDevice);

    // cdf
    // cudaMemcpy(cdf, cdfDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaMemcpy(minimum, minimumDevice, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

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

    printf("\nMIN IS: %ld\n", *minimum);

    // write image (final step)
    stbi_write_jpg(argv[2], width, height, COLOR_CHANNELS, imageOut, 100);

    // free allocated memory
    free(imageIn);
    free(imageOut);
    free(histogram);
    free(cdf);
    free(minimum);
    cudaFree(imageInDevice);
    cudaFree(histogramDevice);
    cudaFree(cdfDevice);
    cudaFree(minimumDevice);

    return 0;
}