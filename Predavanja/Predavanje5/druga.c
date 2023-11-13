#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

// #define N 100000000 // število interaij (100000000 = 100M iteracij)
#define N (4000)
#define T 8 // število niti

struct timespec timeStart, timeEnd;

int main(void) {

    clock_gettime(CLOCK_REALTIME, &timeStart);

    omp_set_num_threads(T);
#pragma omp parallel for schedule(dynamic, 50)
    for (int i = 0; i<N; i++) {
        usleep(5 * i); // vsaka iteracija se dlje časa izvaja in niti niso dobile pošteno razdeljeno delo (nit 3 dela več kot nit 1 oz. čaka dlje)
    } // niti z višjimi indeksi opravijo več dela kot niti z nižjimi indeksi
    // zato nam openmp omogoča, da bolj pošteno razdelimo delo med niti
    // uporabimo "schedule":
    // lahko je static (default), dynamic (dela rezine deložine toliko, kolikor podamo)


    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double timeTaken = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;
    printf("Time taken: %f seconds\n", timeTaken);

    return 0;

}
