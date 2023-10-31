#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 9

pthread_barrier_t barrier; // ne vem, zakaj pravi da tega ne najde, ker se čist normalno izvede

int seznam[] = {1, 5, 2, 0, 7, 4, 9, 8, 3};

// vzame naslova dveh sosednih elementov
void compare_and_swap(int *a, int *b) {

    // če je strogo manjši jih zamenja, sicer jih pusti pri miru
    int tmp;
    if (*b < *a) {
        tmp = *a;
        *a = *b;
        *b = tmp;
    }

}

void sodi_prehod(int *seznam) {

    // povečuje za dve, da zadene sode indekse
    for (int i = 0; i<N - 1; i+=2) {
        compare_and_swap(&seznam[i], &seznam[i + 1]);
    }

}

void lihi_prehod(int *seznam) {

    for (int i = 1; i<N - 1; i+=2) {
        compare_and_swap(seznam + i, seznam + i + 1);
    }

}

void izpisi_seznam(int *seznam) {

    for (int i = 0; i<N; i++) {
        printf("%d ", seznam[i]);
    }

}

int main(void) {

    pthread_barrier_init(&barrier, NULL, N / 2);

    printf("Neurejen seznam:\n");
    for (int i = 0; i<N; i++) {
        printf("%d ", seznam[i]);
    }
    printf("\n");

    // tukaj je N/2, ker bi brez tega oboje 8-krat izvajal
    // še eno težavo (da prehitro končamo, da perskočimo zadnje urejanje) tukaj reši uporaba "<="
    for (int i = 0; i<=N / 2; i++) {
        sodi_prehod(seznam);
        printf("SODI: ");
        izpisi_seznam(seznam);
        printf("\n");

        lihi_prehod(seznam);
        printf("LIHI: ");
        izpisi_seznam(seznam);
        printf("\n");
    }

    return 0;

}