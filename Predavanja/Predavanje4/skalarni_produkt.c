#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N_ELEMENTS 1024 * 1024
#define N_THREADS 4

float *pVecA;
float *pVecB;
float *pVecC;
float dp; // dot product

int main(void) {

    float delniProdukti[N_ELEMENTS];

    pVecA = (float*) malloc(N_ELEMENTS * sizeof(float));
    pVecB = (float*) malloc(N_ELEMENTS * sizeof(float));
    
    // omp_set_num_threads(4);

    // zapisi vrednosti za vektorja
#pragma omp parallel for // parallelno izvedi zanko
    for (int i = 0; i<N_ELEMENTS; i++) {
        pVecA[i] = 1.0;
        *(pVecB + i) = 2.0;
    }

    omp_set_num_threads(N_THREADS);
#pragma omp parallel for
    // #pragma omp parallel for private(dp) 
    // parallelno izvedi zanko, vendar je posledično rezultat narobe
    // lahko uporabimo PRIVATE(), kjer naštejemo
    // te problematične spremenljivke, vendar ker je to sedaj
    // kreirano kot lokalna spremenljivka za vsako nit
    // te vrednosti ne vrača in je posledično rezultat 0.

    // problem je sedaj, da je dp skupna več nitim, ker
    // je globalna
    for (int i = 0; i<N_ELEMENTS; i++) {
        delniProdukti[omp_get_thread_num()] += dp + (*(pVecA + i) * pVecB[i]);
    }

    // izpiši delne produkte
    for (int i = 0; i<N_THREADS; i++) {
        printf("Delni produkti[%d]: %f\n", i, delniProdukti[i]);
        dp += delniProdukti[i];
    }

    printf("Skalarni produkt je: %f\n", dp / 1024.0f); // tole je isto kakor: dp / (float) 1024.0f

    return 0;

}