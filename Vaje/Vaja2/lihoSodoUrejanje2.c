#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

/*
How to compile me:
    - to check if it is sorted and use N random numbers:
      gcc -o lihoSodoUrejanje2 lihoSodoUrejanje2.c -lpthread -O2 -D CHECK_SORTED -D RANDOM_NUMBERS
    - to check if it is sorted and use N random numbers and print array:
      gcc -o lihoSodoUrejanje2 lihoSodoUrejanje2.c -lpthread -O2 -D CHECK_SORTED -D RANDOM_NUMBERS -D PRINT_ARRAY
    - to check if it is sorted and print array (using 10 predefined values)
      gcc -o lihoSodoUrejanje2 lihoSodoUrejanje2.c -lpthread -O2 -D CHECK_SORTED -D PRINT_ARRAY
    - the -D CHECK_SORTED is always recomended to see if the array is sorted correctly
*/

/*
Tested with:
    srun --reservation=psistemi --cpus-per-task=4 lihoSodoUrejanje2
    N = 100000, T = 8, time taken = 5.469577  seconds
    N = 200000, T = 8, time taken = 22.729762 seconds
    N = 300000, T = 8, time taken = 51.384202 seconds
    N = 400000, T = 8, time taken = 91.537148 seconds
*/

#define T 8

typedef struct {
    uint32_t threadID,
             startIndex,
             step;
} threadArgs;

struct timespec timeStart, timeEnd;

#ifdef RANDOM_NUMBERS
void fillArray(int *);
#endif
void *evenOddSort(void *);
#ifdef PRINT_ARRAY
void printArray(int *);
#endif
void compareAndSwap(int *, int *);
#ifdef CHECK_SORTED
int isSorted(int *);
#endif

pthread_barrier_t barrier;
pthread_t threads[T];
threadArgs threadArguments[T];
#ifdef RANDOM_NUMBERS
#define N 400000 // change me, if using RANDOM_NUMBERS flag
int numbers[N];
#else
#define N 10
int numbers[N] = {7, 4, 3, 6, 5, 2, 8, 9, 1, 0};
#endif

int main(void) {

    printf("Sorting %d numbers\n", N);

    uint32_t start = 0;
    #ifdef RANDOM_NUMBERS
    fillArray(numbers);
    #endif

    #ifdef PRINT_ARRAY
    printf("Before sorting:\n");
    printArray(numbers);
    #endif

    pthread_barrier_init(&barrier, NULL, T);

    clock_gettime(CLOCK_REALTIME, &timeStart);

    for (int i = 0; i<T; i++) {
        threadArguments[i].threadID = i;
        if (i < T - 1) {
            threadArguments[i].step = N / T;
        } else {
            threadArguments[i].step = (N / T) + (N % T) - 1;
        }
        threadArguments[i].startIndex = start;
        pthread_create(
            &threads[i],
            NULL,
            evenOddSort,
            (void *) &threadArguments[i]
        );
        start += (threadArguments[i].step * 2);
    }

    for (int i = 0; i<T; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double timeTaken = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;

    pthread_barrier_destroy(&barrier);

    printf("Time taken to sort: %f seconds\n", timeTaken);

    #ifdef PRINT_ARRAY
    printf("After sorting:\n");
    printArray(numbers);
    #endif

    #ifdef CHECK_SORTED
    printf("Array %s sorted\n", isSorted(numbers) ? "IS" : "IS NOT");
    #endif

    return 0;

}

#ifdef PRINT_ARRAY
void printArray(int *arr) {

    for (int i = 0; i<N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

}
#endif

#ifdef RANDOM_NUMBERS
void fillArray(int *arr) {

    srand(time(NULL));
    for (int i = 0; i<N; i++) {
        arr[i] = rand() % N;
    }

}
#endif

void compareAndSwap(int *a, int *b) {

    int tmp;
    if (*b < *a) {
        tmp = *a;
        *a = *b;
        *b = tmp;
    }

}

void *evenOddSort(void *args) {

    int index;
    threadArgs *values = (threadArgs *) args;
    for (int i = 0; i<N; i++) {
        #ifdef PRINT_ARRAY
        printArray(numbers);
        #endif
        for (int j = 0; j<values->step; j++) {
            index = (values->startIndex + (i % 2)) + j * 2; // + 1 or 0 if odd or even, then get the right offset
            if (index + 1 >= N) {
                break;
            }
            compareAndSwap(&numbers[index], &numbers[index + 1]);
        }
        pthread_barrier_wait(&barrier);
    }

}

#ifdef CHECK_SORTED
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
#endif