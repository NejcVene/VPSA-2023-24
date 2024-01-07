#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 1 // koliko kanalov ima slika

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1

#define err(msg, reason) { fprintf(stderr, "Error: %s %s\n", msg, reason); exit(1); }

void CalculateHistogram(unsigned char* image, int width, int height, unsigned long* histogram);
void CalculateCDF(unsigned long* histogram, unsigned long* cdf);
void Equalize(unsigned char * image_in, unsigned char * image_out, int width, int height, unsigned long* cdf);
unsigned char Scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize);
unsigned long findMin(unsigned long* cdf);

int main(int argc, char **argv) {

    if (argc < 2) {
        err("To few arguments", "Usage: <image_in> <image_out>")
    }

    // 0. Najprej naloži sliko
    // Read image from file
    int width, height, cpp;
    // read only DESIRED_NCHANNELS channels from the input image:
    // prebere sliko, pove njene dimenizije, pa zadeve o barvih
    unsigned char *imageIn = stbi_load(argv[1], &width, &height, &cpp, DESIRED_NCHANNELS);
    if(imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);


    //Allocate memory for raw output image data, histogram, and CDF 
	// tole je izhodna slika (velikost eneka kot vhodna)
    unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned long));
    // naredi prostor za histogram
    unsigned long *histogram= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));
    
    unsigned long *CDF= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));

    // 1. Izračun histograma
    CalculateHistogram(imageIn, width, height, histogram);
    // po izvedbi te funkcije imamo napolnjen histogram

    // 2. Izračunaj CDF
    // Namig: Link na 3.1 (parallel prefix sum neki neki link)
    CalculateCDF(histogram, CDF);

    // 3. Poračunaj nove barve po tisti formuli
    Equalize(imageIn, imageOut, width, height, CDF);

    stbi_write_jpg(argv[2], width, height, DESIRED_NCHANNELS, imageOut, 100);


    //Free memory
    // sprosti pomn.
	free(imageIn);
    free(imageOut);
    free(histogram);
    free(CDF);

    return 0;

}

void CalculateHistogram(unsigned char* image, int width, int height, unsigned long* histogram){
    
    //Clear histogram:
    for (int i=0; i<GRAYLEVELS; i++) {
        histogram[i] = 0;
    }
    
    // pojdi skozi sliko (slika je linerano polje)
    //Calculate histogram
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            // tole vrne vrednost piksla i,j in to določa piksel v histo.
            histogram[image[i*width + j]]++;
        }
    }

    int sumOfPixles = 0;
    printf("GRAYLEVELS: %d\n", GRAYLEVELS);
    // print some histo. values for testing
    for (int i = 0; i<GRAYLEVELS; i++) {
        sumOfPixles += histogram[i];
        printf("Color %d has %ld pixels\n", i, histogram[i]);
    }
    printf("Sum of pixels: %d\nHEIGHT x WIDTH %d\nSAME: %d\n\n", 
            sumOfPixles, height * width, sumOfPixles == height * width);

    /*
    V CUDA je tole treba narediti atomično
    Namig: Cuda by example.
    */
}

void CalculateCDF(unsigned long* histogram, unsigned long* cdf){
    
    // clear cdf:
    for (int i=0; i<GRAYLEVELS; i++) {
        cdf[i] = 0;
    }
    
    // calculate cdf from histogram
    cdf[0] = histogram[0]; // postavi na 0 vse elemenet (enako kot histo.)
    for (int i=1; i<GRAYLEVELS; i++) {
        // prvi el. je enak koliko px. ima barvo 0
        // koliko px ima barvo 2? Tolk kot barvo 1 in 0 + tolk kot barvo 2
        // koliko px. ima barvo 3 Tolk kot barvo 0 in 1 in 2 + tolk kot barvo 3
        // cdf[3] = h[0] + h[1] + h[2] + h[3]
        // cdf[4] = h[0] + h[1] + h[2] + h[3] + h[4]
        // cdf[5] = h[0] + h[1] + h[2] + h[3] + h[4] + h[5]
        cdf[i] = cdf[i-1] + histogram[i];
        // za CUDA rabmo še eno notranjo for zanko
        // nit 0 ima 0 dela
        // nit 256 ima 256 seštevanj
        // to je znano kot prefix sum
        // kak zdej to naredit bolj učinkovito (to je na tistem linku)
    }

    printf("CDF:\n");
    for (int i = 0; i<GRAYLEVELS; i++) {
        printf("How many px have color %d? %ld\n", i, cdf[i]);
    }

}

void Equalize(unsigned char * image_in, unsigned char * image_out, int width, int height, unsigned long* cdf){
     
    unsigned long imageSize = width * height;
    
    unsigned long cdfmin = findMin(cdf);

    printf("\nMIN IS: %ld\n", cdfmin);
    
    //Equalize: namig: blok niti naj si CDF naloži v skupni pomnilnik
    // pojdi čez sliko in računaj piksle
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            image_out[(i*width + j)] = Scale(cdf[image_in[i*width + j]], cdfmin, imageSize);
        }
        // scale je tista formula
    }
}

unsigned char Scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize){
    
    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = round(scale * (float)(GRAYLEVELS-1)); // ulomek množi s 255 in zaokroži
    
    return (int)scale;

    // minCDF je prva ne ničelna vrednost
    // kak najdit?
    // 1. skozi cdf s zanko in preveri vrednosti. Ko najdše vrednost return.
    // 2. Zanka se sprehaja dokler ne pridemo do konca in dokler cdf != 0
    // findMin to implementira
}

unsigned long findMin(unsigned long* cdf){
    
    unsigned long min = 0;
    // grem skozi CDF dokler ne najdem prvi nenicelni element ali pridem do konca
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
    }
    
    return min;
}