#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
vidimo, da je tale rešitev počasnješa, ker uporablja
zaklepanje. Torej ne uporabljaj zaklepanja, če res
ni treba oz. to rešimo s lastnimi spremenjivkami.
*/

#define N_ELEMENTS 1024 * 1024 * 32
#define N_THREADS 12
#define LOCKING

float *pVecA;
float *pVecB;
float *pVecC;
float dp; // dot product

int main(void) {

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

#ifdef LOCKING
#pragma omp parallel for
    for (int i = 0; i<N_ELEMENTS; i++) {
        #pragma omp critical // tole sedaj popravi našo težavo
        { // <-- zakleni ključavnico
            dp = dp + (*(pVecA + i) * pVecB[i]);
        } // <-- odkleni ključavnico
    }
#endif

    printf("Skalarni produkt je: %f\n", dp / 1024.0f); // tole je isto kakor: dp / (float) 1024.0f

    return 0;

}