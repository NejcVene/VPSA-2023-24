// serijska izvedba iskanja prijateljskega števila

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000000

int vsotaDeljiteljev(int);

int main(void) {

    int sumX = 0, sumY = 0;
    for (int x = 1; x<N; x++) {
        sumX = vsotaDeljiteljev(x);
        for (int j = x + 1; j<x; j++) {
            sumY = vsotaDeljiteljev(j);
            if (x == sumY && j == sumX) {
                // is a friend
            }
        }
    }

    return 0;

}

int vsotaDeljiteljev(int x) {

    int vsota = 0;
    for (int i = 2; i<x; i++) {
        if (x % i) {
            vsota += i + x / i;
        }
    }
    
    return vsota;

}