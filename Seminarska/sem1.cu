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

int main(int argc, char **argv){

    // check for arguments
    if (argc < 2) {
        err("To few arguments.", "Usage: <image_in> <image_out>")
    }

    // host variables
    int width, height, imageSizeCuda, cpp = 1;
    unsigned char *imageIn, *imageOut;
    unsigned int *histogram, *cdf;
    size_t imageSize;

    // device variables
    unsigned char *imageInDevice;
    unsigned int *histogramDevice;

    // load image from file
    if (!(imageIn = stbi_load(argv[1], &width, &height, &cpp, 1))) {
        err("stbi_load", "Could not load image")
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", argv[1], width, height, cpp);

    // allocate memory for histogram
    if (!(histogram = (unsigned int *) calloc(GRAY_LEVELS, sizeof(unsigned int)))) { // SET IT TO ZERO
        err("malloc", "Could not allocate memory for histogram on host")
    }

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

    // copy data to device
    // image
    cudaMemcpy(imageInDevice, imageIn, imageSize, cudaMemcpyHostToDevice);
    // histogram
    cudaMemcpy(histogramDevice, histogram, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // run calculateHistogram
    dim3 blockSize(256, 1, 1);
    dim3 gridSize(1024 / 256, 1, 1);
    

    // int gridDim = (imageSizeCuda + 1) / BLOCK_SIZE;

    calculateHistogram<<<gridSize,blockSize>>>(imageInDevice, imageSizeCuda, histogramDevice);

    // copy result from device to host
    // histogram
    cudaMemcpy(histogram, histogramDevice, GRAY_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int sumOfPixles = 0;
    printf("GRAYLEVELS: %d\n", GRAY_LEVELS);
    for (int i = 0; i<GRAY_LEVELS; i++) {
        sumOfPixles += histogram[i];
        printf("Color %d has %ld pixels\n", i, histogram[i]);
    }
    printf("Sum of pixels: %d\nHEIGHT x WIDTH %d\nSAME: %d\n\n",
            sumOfPixles, height * width, sumOfPixles == height * width);

    // write image (final step)
    stbi_write_jpg(argv[2], width, height, COLOR_CHANNELS, imageOut, 100);

    // free allocated memory
    free(imageIn);
    free(imageOut);
    cudaFree(imageInDevice);
    cudaFree(histogramDevice);

    return 0;
}