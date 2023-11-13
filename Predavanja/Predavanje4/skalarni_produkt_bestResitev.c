#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N_ELEMENTS 1024 * 1024 * 32
#define N_THREADS 12

struct timespec timeStart, timeEnd;

float *pVecA;
float *pVecB;
float *pVecC;
float dp; // dot product

int main(void) {

    pVecA = (float*) malloc(N_ELEMENTS * sizeof(float));
    pVecB = (float*) malloc(N_ELEMENTS * sizeof(float));
    
    // omp_set_num_threads(4);

    clock_gettime(CLOCK_REALTIME, &timeStart);

    // zapisi vrednosti za vektorja
#pragma omp parallel for // parallelno izvedi zanko
    for (int i = 0; i<N_ELEMENTS; i++) {
        pVecA[i] = 1.0;
        *(pVecB + i) = 2.0;
    }

    omp_set_num_threads(N_THREADS);

#pragma omp parallel for reduction(+:dp) // tole zdej naredimo tisto po sliki (jih reducira in končni rezultat vpiše v dp)
    for (int i = 0; i<N_ELEMENTS; i++) {
        dp = dp + (*(pVecA + i) * pVecB[i]);
    }

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double timeTaken = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;

    printf("Skalarni produkt je: %f\n", dp / 1024.0f); // tole je isto kakor: dp / (float) 1024.0f
    printf("Time taken to calculate DP: %f seconds\n", timeTaken);

    return 0;

}