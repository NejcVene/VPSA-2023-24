#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

/*
https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 4
#define BSX 16
#define BSY 16


__device__ unsigned char vrniPiksel(unsigned char *imgIn, int row, int col, int channel,
                        int width, int height, int cpp){

    if(row < 0 || row > height-1){
        return 0;
    }
    else if (col < 0 || row > width-1){
        return 0;
    }
    else {
        return imgIn[(row*width + col)*cpp + channel];
    }
    
}

__global__ void makeHisto(unsigned char *buffer, long size, unsigned int *histo) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x,
        stride = blockDim.x * gridDim.x;
 
    while (i < size) {
        atomicAdd(&(histo[buffer[i]]), 1);
        i += stride;
    } 

}

int main(int argc, char *argv[]){

    if (argc < 3)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    unsigned int *devHisto;

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", szImage_in_name, width, height, cpp);
    //ne glede na dejansko število kanalov bomo vedno predpostavili 4 kanale:
    cpp = 1; // 1 color channel, ker je prof. neki reku o tem ???

    // rezerviraj prostor v pomnilniku za izhodno sliko:
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);


    // TODO: vaja
    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // rezerviraj prostor na napravi za obe sliki:
    cudaMalloc(&d_imageIn, datasize);

    cudaMalloc((void **) &devHisto, datasize * sizeof(long));

    cudaMemset(devHisto, 0, datasize * sizeof(int));

    // prenesimo vhodno sliko na GPE:
    cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice);

    dim3 blockSize(BSX, BSY);
    dim3 gridSize(ceil(width/BSX), ceil(height/BSY));
    
    // pokliči kernel z izvajalnim okoljem:
    makeHisto<<<gridSize,blockSize>>>(d_imageIn, datasize, );

    unsigned int histo[datasize];
    // prenesimo izhodno sliko iz GPE na gostitelja:
    cudaMemcpy(histo, devHisto, datasize * sizeof(int), cudaMemcpyDeviceToHost);

    // shranimo izhodno sliko
    long histoCount = 0;
    for (int i=0; i<datasize; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum: %ld\n", histoCount );


    return 0;
}