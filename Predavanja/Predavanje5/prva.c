// Računanje π.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

// #define N 100000000 // število interaij (100000000 = 100M iteracij)
#define N (1024 * 1024 * 64)
#define T 8 // število niti

struct timespec timeStart, timeEnd;

double dx = 1.0 / (double) N,
       fy,
       pi = 0,
       x;

int main(void) {

    clock_gettime(CLOCK_REALTIME, &timeStart);

    omp_set_num_threads(T);
#pragma omp parallel for reduction(+:pi)
    for (int i = 0; i<N; i++) {
        double x = (double) i * dx; // za vsako nit je privat
        double fy = sqrt(1.0 - x * x); // za vsako nit je privat
        pi += fy * dx;
    }
    pi *= 4.0;

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double timeTaken = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;
    printf("Time taken: %f seconds\n", timeTaken);
    printf("PI pri %d iteracijah = %f z %d nitmi\n", N, pi, T);

    return 0;

}

/* Iterativno
for (int i = 0; i<N; i++) {
        x = (double) i * dx;
        fy = sqrt(1.0 - x * x);
        pi += fy * dx;
    }
    pi *= 4.0;
*/
/*
Vidimo, da vse niti pišejo v x, fy in pi. To ni good.
*/