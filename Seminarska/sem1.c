#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define err(msg, reason) { fprintf(stderr, "Error: %s %s\n", msg, reason); exit(1); }
#define COLOR_CHANNELS 1
#define GRAY_LEVELS 256

int main(int argc, char **argv){

    if (argc < 3) {
        err("To few arguments.", "Usage: <image_in> <image_out>")
    }

    int width, height, cpp = 1;
    unsigned char *imageIn, *imageOut;
    unsigned long *histogram, *cdf;
    size_t imageSize;

    // load image from file
    if (!(imageIn = stbi_load(argv[1], &width, &height, &cpp, COLOR_CHANNELS))) {
        err("stbi_load", "Could not load image")
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", argv[1], width, height, cpp);

    // calculate needed size of memory and allocate it
    imageSize = width * height * cpp * sizeof(unsigned long);
    if (!(imageOut = (unsigned char *) malloc(imageSize))) {
        err("malloc", "Could not allocate memory for output image on host")
    }

    // write image (final step)
    stbi_write_jpg(argv[2], width, height, COLOR_CHANNELS, imageOut, 100);

    // free allocated memory
    free(imageIn);
    free(imageOut);

    return 0;
}