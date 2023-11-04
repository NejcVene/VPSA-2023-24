/*
1. Struktura, ki hrani ID vsake niti
2. Ena funkcija, ki jo vse niti uporabljajo. Znotraj
   te funkcije, se preveri, ali se naredi sodi ali lihi korak.
   Glede na to se izvede urejanje s ustreznimi indeksi.
3. Vse niti hranimo v tabeli velikosti T.
4. T, ki definira število niti dobimo kot argument argv[1]
5. Števila, ki jih urejamo hranimo v tabeli velikosti N.
6. N, ki definira velikost tabele števil, ki jih moramo
   urediti dobimo kot argument argv[2].
*/

// local: 1:56:56
// arnes: 00:51:47

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>

#define N 10
#define T 8

#define err(msg) { fprintf(stderr, "Error: %s\n", msg); exit(1); }

void createNumbers(int *);
void printArray(int *);
void *evenOddSort(void *); // apperenty this is also known as Brick sort
bool compareAndSwap(int *, int *);

struct timespec timeStart, timeEnd;

typedef struct {
    uint32_t threadID,
             startIndex,
             step;
    pthread_t thread;
} threadArgs;

pthread_barrier_t barrier;

int even = 1;
int numbers[N];
bool changedMade[T] = {false};

bool is_sorted() {
    int prev = numbers[0];
    for (int i = 1; i < N; ++i) {
        if (prev > numbers[i]) return false;
        prev = numbers[i];
    }
    return true;
}

int main(int argc, char **argv) {

    int start = 0;
    threadArgs threads[T];

    createNumbers(numbers);
    printArray(numbers);

    pthread_barrier_init(&barrier, NULL, T);

    clock_gettime(CLOCK_REALTIME, &timeStart);

    for (int i = 0; i<T; i++) {
        threads[i].threadID = i;
        threads[i].startIndex = start;
        threads[i].step = (i + 1) * (N / 2) / T - i * (N / 2) / T;
        pthread_create(&threads[i].thread, 
                        NULL, 
                        evenOddSort, 
                        (void *) &threads[i]);
        start += (threads[i].step * 2);
    }

    for (int i = 0; i<T; i++) {
        pthread_join(threads[i].thread, NULL);
    }

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double elapsed_time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;

    pthread_barrier_destroy(&barrier);

    printf("Result\n");
    // printArray(numbers);
    if (is_sorted()) {
        printf("Stevilke so urejene pravilno! :)");
    } else {
        printf("Stevilke niso urejene pravilno! :(");
    }
    printf("\n");
    printf("Čas izvajanja: %f sekund \n", elapsed_time);

    return 0;

}

void createNumbers(int *arr) {

    srand(time(NULL));
    for (int i = 0; i<N; i++) {
        arr[i] = rand() % N;
    }

}

void printArray(int *arr) {

    for (int i = 0; i<N; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

}

void *evenOddSort(void *args) {

    int index;
    threadArgs *values = (threadArgs *) args;
    // changedMade[values->threadID] = false;
    // printf("ID: %d StartIndex: %d EndIndex: %d\n", values->threadID, values->startIndex, values->step);
    // printf("Pozdrav s niti %d\n", values->threadID);
    for (int i = 0; i<=N; i++) {
        // printf("Sorting is %s\n", even ? "Even" : "Odd");
        // printf("RUNNING THREAD: %d\n", values->threadID);
        for (int j = 0; j<values->step; j++) {
            index = (values->startIndex + (i % 2)) + j * 2;
            // printf("Index: %d\n", index);
            changedMade[values->threadID] = compareAndSwap(&numbers[index], &numbers[index + 1]);
        }
        printf("Tread %d made changes? %s\n", values->threadID, changedMade[values->threadID] ? "True" : "False");
        printArray(numbers);
        pthread_barrier_wait(&barrier);

        bool isChange = false;
        for (int j = 0; j<T; j++) {
            if (changedMade[j]) {
                isChange = true;
                break;
            }
        }

        if (!isChange) {
            return NULL;
        }

    }

}

bool compareAndSwap(int *a, int *b) {

    // printf("COMPARING: %d %d\n", *a, *b);
    int tmp;
    if (*b < *a) {
        tmp = *a;
        *a = *b;
        *b = tmp;
        return true;
    }
    return false;

}