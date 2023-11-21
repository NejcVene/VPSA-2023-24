#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define N 1000000 // 67 za 1024*1024*4, 40 za 1000000

int getSumOfDivisor(int);
void multithreadAmicable(int);

int sumOfDivisors[N];

int main(void) {

    printf("Size of N = %d\n", N);

    multithreadAmicable(32);

    return 0;

}

void multithreadAmicable(int numOfThreads) {

    int finalCount = 0, finalSum = 0, sdi;
    double timeStart, timeEnd, timeTaken;

    omp_set_num_threads(numOfThreads);

    timeStart = omp_get_wtime();

#pragma omp parallel for schedule(dynamic, 50)
    for (int x = 1; x<N; x++) {
        sumOfDivisors[x] = getSumOfDivisor(x);
    }

#pragma omp parallel for reduction(+:finalCount, finalSum)
    for (int i = 1; i<N; i++) {
        sdi = sumOfDivisors[i];
        if (sdi < N) {
            if (i < sdi) {
                if (sumOfDivisors[sdi] == i) {
                    finalCount += 1;
                    finalSum += (i + sdi);
                }
            }
        }
    }

    timeEnd = omp_get_wtime();
    timeTaken = timeEnd - timeStart;

    printf("(%s) Time taken for %d threads: %f seconds. Count of amicable number couples: %d. SUM: %d\n",
            numOfThreads > 1 ? "Parallel" : "Serial",
            numOfThreads,
            timeTaken,
            finalCount,
            finalSum);

}

int getSumOfDivisor(int x) {

    int sum = 1;
    for (int i = 2; i<=sqrt(x); i++) {
        if (x % i == 0) {
            sum += i;
            if (x / i != i) {
                sum += x / i;
            }
        }
    }
    
    return sum;

}

// compiled with: gcc -o paralelno prijateljskaStevila.c -O2 -fopenmp -lm -Wall
// ran with: srun --reservation=psistemi --cpus-per-task=4 paralelno
// Rezultati so za N = 1000000
/*

S = Ts / Tp
Ts -> čas sekvenčnega algoritma
Tp -> čas paralelnega algoritma

schedule(dynamic, 50):
    Ts  = 7.697296 s | 
    T2  = 1.956837 s | S = 3.9335
    T4  = 1.921286 s | S = 4.0063
    T8  = 0.961740 s | S = 8.0035
    T16 = 0.966406 s | S = 7.9648
    T32 = 0.965267 s | S = 7.9742

schedule(static):
    Ts  = 7.697296 s | 
    T2  = 2.545284 s | S = 3.0241
    T4  = 2.185815 s | S = 3.5214
    T8  = 2.024369 s | S = 3.8023
    T16 = 1.959428 s | S = 3.9283
    T32 = 1.928419 s | S = 3.9915

schedule(guided, 3):
    Ts  = 7.697296 s | 
    T2  = 1.958828 s | S = 3.9295
    T4  = 1.923905 s | S = 4.0008
    T8  = 1.927130 s | S = 3.9941
    T16 = 1.923845 s | S = 4.0009
    T32 = 1.923635 s | S = 4.0014

Glede na pridobljene rezultate, najboljšo pohitritev poda uporaba schedule(dynamic, 50), in sicer s uporabo 8 niti
je S = 8.0035, kar je najboljši od vseh.

*/
