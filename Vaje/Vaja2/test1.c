#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    int start, id, par;
} nit;

void sort(nit *niti, int N);

int main(void) {

    int N = 8, T = 2, ti, n = 0;
    nit niti[T];
    for (int i = 0; i<T; i++) {
        niti[i].id = i;
        niti[i].start = n;
        niti[i].par = (i + 1) * (N / 2) / T - i * (N / 2) / T;
        n = n + niti[i].par * 2;
        printf("ID: %d Start: %d Par: %d\n", niti[i].id, niti[i].start, niti[i].par);
    }
    sort(niti, N);

    return 0;

}

void sort(nit *niti, int N) {

    int start;
    for (int i = 0; i<N; i++) {
        start = niti->start + (i % 2);
        printf("START: %d (%d %d)\n", start, niti->id, niti->start);
    }

}