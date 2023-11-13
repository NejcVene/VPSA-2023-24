#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#define T 4

#ifdef RANDOM_NUMBERS
#define N 1000 // change me, if using RANDOM_NUMBERS flag
int numbers[N];
#else
#define N 9
int numbers[N] = {7, 4, 3, 6, 5, 2, 8, 9, 1};
#endif

void* funkcija_niti(void *);

typedef struct {
    int threadID,
        indexStart,
        blockSize;
} threadArgs;
struct timespec timeStart, timeEnd;

pthread_barrier_t barrier;
pthread_t threads[T];
threadArgs treadArguments[T];
unsigned int *pSeznam = numbers; // naslov začetnega elementa v seznamu numbers
void* (*pFunkcijaNiti)(void*) = funkcija_niti;
bool globalSorted = true;
pthread_mutex_t lock;

void izpisi_seznam(unsigned int *pSeznam){
    for (int i = 0; i < N; i++)
    {
        printf("%d ", *(pSeznam+i));
    }
    printf("\n");
}

bool primerjaj_in_zamenjaj(int *a, int *b) {

    bool sorted = true;
    unsigned int tmp;
    if (*b < *a) {
        tmp = *a;
        *a = *b;
        *b = tmp;
        sorted = false;
    }
    return sorted;

}

bool sodi_prehod(int iEnd, unsigned int* pSeznam){
    // printf("Sodi prehod: %d\n", *pSeznam);
    bool sorted = true;
    for (int i = 0; i < iEnd; i=i+2){
        sorted &= primerjaj_in_zamenjaj(&pSeznam[i], &pSeznam[i+1]);
    }
    // printf("\n");
    return sorted;
}

bool lihi_prehod(int iEnd, unsigned int* pSeznam){
    bool sorted = true;
    for (int i = 1; i < iEnd; i=i+2){
        sorted &= primerjaj_in_zamenjaj(&pSeznam[i], &pSeznam[i+1]);
    }
    return sorted;
}

void fillArray(int *arr) {

    srand(time(NULL));
    for (int i = 0; i<N; i++) {
        arr[i] = rand() % N;
    }

}

int isSorted(int *arr) {

    if (N == 0 || N == 1) {
        return 1;
    }
    for (int i = 1; i<N; i++) {
        if (arr[i - 1] > arr[i]) {
            return 0;
        }
    }
    return 1;

}

int main(void) {

    #ifdef RANDOM_NUMBERS
    fillArray(numbers);
    #endif

    clock_gettime(CLOCK_REALTIME, &timeStart);

    pthread_barrier_init(&barrier, NULL, T);

    for(int i = 0; i < T; i++){
        treadArguments[i].threadID = i;
        treadArguments[i].indexStart = i * (N / T);
        if (i < T - 1) {
            treadArguments[i].blockSize = N / T;
        } else {
            treadArguments[i].blockSize = N / T + (N % T); // N % T da dobimo ostanek pri deljenju, ki ga potem prištejemo celemu število, da dobimo velikost bloka za zadnjo nit
        }
        pthread_create(&threads[i], NULL, pFunkcijaNiti, (void *) &treadArguments[i]);
    }

    for (int i = 0; i<T; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_barrier_destroy(&barrier);

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double timeTaken = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;
    printf("Time taken: %f seconds\n", timeTaken);
    // izpisi_seznam(numbers);
    printf("Array %s sorted\n", isSorted(numbers) ? "IS" : "IS NOT");

    return 0;

}

void *funkcija_niti(void *args) {

    bool sorted = true;
    threadArgs *values = (threadArgs *) args;
    printf("Doing something ...\n");
    for (int i = 0; i<N; i++) {
        sorted &= sodi_prehod(values->blockSize, pSeznam + values->indexStart);
        pthread_mutex_lock(&lock);
        globalSorted &= sorted;
        pthread_mutex_unlock(&lock);

        pthread_barrier_wait(&barrier);
        
        sorted &= lihi_prehod(values->blockSize, pSeznam + values->indexStart);
        pthread_mutex_lock(&lock);
        globalSorted &= sorted;
        pthread_mutex_unlock(&lock);
        pthread_barrier_wait(&barrier);

        if (globalSorted == true) {
            break;
        }
        if (values->threadID == 0) {
            globalSorted = true;
        }

        pthread_barrier_wait(&barrier);
        
    }

}

/*
Tukaj bi se lahko dalo omptimizirati tako, da
preverjamo, če se je kaj spremenilo. To delovo v
compareAndSwap pri čemer uporabimo neke vrste flag. Ta flag je
potrebno skiri za ključavnico.
Lahko pa naredmo tudo tako, da ima vsaka nit ta flag kot svojo lokalno spremenljivko
v argumentih. Nato pa omogočimo samo eni niti, da preveri za vse ostale, če se je ta flag
kje spremnil, recimo to dela nit z ID 0. To preverimo pred barriero (se mi zdi, da imamo sedaj dve?)
*/