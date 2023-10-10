#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_OF_ELEMENTS 1024 * 1024

void* funkcija_niti(void *);

typedef struct {
    unsigned int thread_id;
    float *vecA, *vecB, *vecC;  
} argumenti_t;

pthread_t nit1, nit2;
argumenti_t args1, args2;

int main(void) {
    
    float *vecA, *vecB, *vecC;
    // dinamično ustvarimo tri vektorje v pomn.
    vecA = (float*) malloc(sizeof(float) * NUM_OF_ELEMENTS);
    vecB = (float*) malloc(sizeof(float) * NUM_OF_ELEMENTS);
    vecC = (float*) malloc(sizeof(float) * NUM_OF_ELEMENTS);

    // vektorja A in B incializiramo na neke vrednsti (A na 2.0, B na 3.0)
    for (int i = 0; i<NUM_OF_ELEMENTS; i++) {
        vecA[i] = 2.0;
        vecB[i] = 3.0;
    }

    args1.thread_id = 0;
    args1.vecA = vecA;
    args1.vecB = vecB;
    args1.vecC = vecC;

    args2.thread_id = 1;
    args2.vecA = vecA;
    args2.vecB = vecB;
    args2.vecC = vecC;

    pthread_create(&nit1, NULL, funkcija_niti, (void *) &args1);
    pthread_create(&nit2, NULL, funkcija_niti, (void *) &args2);

    // preverimo rezultat
    float rezultat = 0.0;
    for (int i = 0; i<NUM_OF_ELEMENTS; i++) {
        rezultat += vecC[i];
    }

    printf("Rezultat je: %.2f\n", rezultat);

    pthread_join(nit1, NULL);
    pthread_join(nit2, NULL);


    // sprosti pomn. ki smo ga rezervirali
    free(vecA);
    free(vecB);
    free(vecC);

    return 0;

}

void* funkcija_niti(void *arg) {

  // za dostop do elementov je navaden void naslov, zato moramo
  // definirati nov kazalec na strukturo in ga pretovirit v ta tip.
  argumenti_t *argumenti = (argumenti_t *) arg;
  for (int i = argumenti->thread_id * (NUM_OF_ELEMENTS / 2); i<(argumenti->thread_id + 1) * (NUM_OF_ELEMENTS / 2); i++) {
    argumenti->vecC[i] = argumenti->vecA[i] + argumenti->vecB[i];
  }

}