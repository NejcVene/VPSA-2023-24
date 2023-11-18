#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

/*
How to compile me:
    - to check if it is sorted and use N random numbers:
      gcc -o lihoSodoUrejanje3 lihoSodoUrejanje3.c -lpthread -O2 -D CHECK_SORTED -D RANDOM_NUMBERS
    - to check if it is sorted and use N random numbers and print array:
      gcc -o lihoSodoUrejanje3 lihoSodoUrejanje3.c -lpthread -O2 -D CHECK_SORTED -D RANDOM_NUMBERS -D PRINT_ARRAY
    - to check if it is sorted and print array (using 10 predefined values)
      gcc -o lihoSodoUrejanje3 lihoSodoUrejanje3.c -lpthread -O2 -D CHECK_SORTED -D PRINT_ARRAY
    - the -D CHECK_SORTED is always recomended to see if the array is sorted correctly
*/

/*
Tested with:
    srun --reservation=psistemi --cpus-per-task=4 lihoSodoUrejanje3
    N = 1000000, T = 8, time taken = 466.442750  seconds
    N = 1000000, T = 4, time taken = 402.013284  seconds
    N = 1000000, T = 2, time taken = 484.665847  seconds
    N = 1000000, T = 1, time taken = 942.019233  seconds
*/

#define T 1

#ifdef PRINT_ARRAY
void printArray(uint32_t *);
#endif
#ifdef RANDOM_NUMBERS
void generateArray(uint32_t *);
#endif
bool even(uint32_t, uint32_t *, uint32_t);
bool odd(uint32_t, uint32_t *, uint32_t);
bool compareAndSwap(uint32_t *, uint32_t *);
void *sort(void *);
#ifdef CHECK_SORTED
bool checkSorted(uint32_t *);
#endif

typedef struct {
    uint32_t threadID,
            startIndex,
            step;
} threadArgs;

struct timespec timeStart, timeEnd;

#ifdef RANDOM_NUMBERS
#define N 1000000
uint32_t numbers[N];
#else
#define N 10
int numbers[N] = {7, 4, 3, 6, 5, 2, 8, 9, 1, 0};
#endif
pthread_t threads[T];
threadArgs threadArguments[T];
bool globalSorted = true;

pthread_barrier_t barrier;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int main(void) {

    printf("Sorting %d numbers with %d threads\n", N, T);

    #ifdef RANDOM_NUMBERS
    generateArray(numbers);
    #endif

    #ifdef PRINT_ARRAY
    printf("Before sorting:\n");
    printArray(numbers);
    #endif

    clock_gettime(CLOCK_REALTIME, &timeStart);

    pthread_barrier_init(&barrier, NULL, T);

    for (int i = 0; i<T; i++) {
        threadArguments[i].threadID = i;
        threadArguments[i].startIndex = i * (N / T);
        if (i < T - 1) {
            threadArguments[i].step = (N / T);
        } else {
            threadArguments[i].step = (N / T) + (N % T) - 1;
        }
        pthread_create(&threads[i],
                      NULL,
                      sort,
                      (void *) &threadArguments[i]);
    }

    for (int i = 0; i<T; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double timeTaken = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;

    #ifdef PRINT_ARRAY
    printf("After sorting:\n");
    printArray(numbers);
    #endif

    #ifdef CHECK_SORTED
    printf("Array %s\n", checkSorted(numbers) ? "IS sorted" : "IS NOT sorted");
    #endif
    printf("Time taken: %f seconds\n", timeTaken);

    return 0;

}

#ifdef PRINT_ARRAY
void printArray(uint32_t *arr) {

    for (int i = 0; i<N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

}
#endif

#ifdef RANDOM_NUMBERS
void generateArray(uint32_t *arr) {

    srand(time(NULL));
    for (int i = 0; i<N; i++) {
        arr[i] = rand() % 100;
    }

}
#endif

bool even(uint32_t endIndex, uint32_t * arr, uint32_t id) {

    bool isSorted = true;
    uint32_t i = (id * (N / T)) % 2;
    for (;i<endIndex; i+=2) {
        isSorted &= compareAndSwap(&arr[i], &arr[i + 1]);
    }

    return isSorted;

}


bool odd(uint32_t endIndex, uint32_t *arr, uint32_t id) {

    bool isSorted = true;
    uint32_t i = 1 - ((id * (N / T)) % 2);
    for (;i<endIndex; i+=2) {
        isSorted &= compareAndSwap(&arr[i], &arr[i + 1]);
    }

    return isSorted;

}

bool compareAndSwap(uint32_t *b, uint32_t *a) {

    bool isSorted = true;
    uint32_t tmp;
    if (*b > *a) {
        tmp = *a;
        *a = *b;
        *b = tmp;
        isSorted = false;
    }

    return isSorted;

}

void* sort(void* args) {

    threadArgs *values = (threadArgs*) args;
    bool isSorted = true;

    for (int i = 0; i < N; i++) {
        isSorted = true;
        isSorted &= even(values->step, numbers + (values->startIndex), values->threadID);

        pthread_mutex_lock(&lock);

        globalSorted &= isSorted;

        pthread_mutex_unlock(&lock);

        pthread_barrier_wait(&barrier);

        isSorted = true;
        isSorted &= odd(values->step, numbers + (values->startIndex), values->threadID);

        pthread_mutex_lock(&lock);

        globalSorted &= isSorted;

        pthread_mutex_unlock(&lock);

        pthread_barrier_wait(&barrier);

        if (globalSorted) {
            break;
        }

        pthread_barrier_wait(&barrier);

        if (values->threadID == 0) {
            globalSorted = true;
        }

        pthread_barrier_wait(&barrier);

    }

    return NULL;

}

#ifdef CHECK_SORTED
bool checkSorted(uint32_t *arr) {

    if (N == 0 || N == 1) {
        return 1;
    }
    for (uint32_t i = 1; i<N; i++) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }

    return true;

}
#endif