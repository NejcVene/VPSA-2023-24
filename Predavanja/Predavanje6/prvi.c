// serijska izvedba iskanja prijateljskega števila

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 6500

int vsotaDeliteljevFunc(int);

int vsoteDeliteljev[N];

int main(void) {

    int sdi;
    for (int x = 1; x<N; x++) {
        vsoteDeliteljev[x] = vsotaDeliteljevFunc(x);
        printf("Vsota deliteljev števila %d je %d\n", x, vsoteDeliteljev[x]);
    }

    for (int i = 1; i<N; i++) {
        sdi = vsoteDeliteljev[i];
        if (sdi < N) {
            if (i < sdi) {
                if (vsoteDeliteljev[sdi] == i) {
                    printf("Prijatelski števili: %d %d\n", i, sdi);
                }
            }
        }
    }

    return 0;

}

int vsotaDeliteljevFunc(int x) {

    int vsota = 1;
    for (int i = 2; i<=sqrt(x); i++) {
        if (x % i == 0) {
            vsota += i;
            if (x / i != i) {
                vsota += x / i;
            }
        }
    }
    
    return vsota;

}