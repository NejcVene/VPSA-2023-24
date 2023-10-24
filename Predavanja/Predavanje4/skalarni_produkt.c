#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N_ELEMENTS 1024 * 1024

float *pVecA;
float *pVecB;
float *pVecC;
float dp; // dot product

int main(void) {

    pVecA = (float*) malloc(N_ELEMENTS * sizeof(float));
    pVecB = (float*) malloc(N_ELEMENTS * sizeof(float));

    // zapisi vrednosti za vektorja
    for (int i = 0; i<N_ELEMENTS; i++) {
        pVecA[i] = 1.0;
        *(pVecB + i) = 2.0;
    }

    // omp_set_num_threads(4);

    for (int i = 0; i<N_ELEMENTS; i++) {
        dp = dp + (*(pVecA + i) * pVecB[i]);
    }

    printf("Skalarni produkt je: %f\n", dp / 1024.0f); // tole je isto kakor: dp / (float) 1024.0f

    return 0;

}