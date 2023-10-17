#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_OF_ELEMENTS 1024 * 1024
#define NUM_OF_THREADS 8

void* funkcija_niti(void *);

typedef struct {
    unsigned int thread_id;
    float *vecStripA, *vecStripB, *vecStripC;  
} argumenti_t;

pthread_t niti[NUM_OF_THREADS];
argumenti_t argumenti[NUM_OF_THREADS];

int main(void) {
    
    float *vecA, *vecB, *vecC;
    // dinamično ustvarimo tri vektorje v pomn.
    vecA = (float*) malloc(sizeof(float) * NUM_OF_ELEMENTS);
    vecB = (float*) malloc(sizeof(float) * NUM_OF_ELEMENTS);
    vecC = (float*) malloc(sizeof(float) * NUM_OF_ELEMENTS);

    // vektorja A in B incializiramo na neke vrednsti (A na 2.0, B na 3.0)
    // tale del se ne da pohitriti
    for (int i = 0; i<NUM_OF_ELEMENTS; i++) {
        vecA[i] = 2.0;
        vecB[i] = 3.0;
    }

    for (int i = 0; i<NUM_OF_THREADS; i++) {
      argumenti[i].thread_id = i;
      argumenti[i].vecStripA = vecA + i * (NUM_OF_ELEMENTS / NUM_OF_THREADS);
      argumenti[i].vecStripB = vecB + i * (NUM_OF_ELEMENTS / NUM_OF_THREADS);
      argumenti[i].vecStripC = vecC + i * (NUM_OF_ELEMENTS / NUM_OF_THREADS);
      pthread_create(&niti[i], NULL, funkcija_niti, (void *) &argumenti[i]);
    }

    for (int i = 0; i<NUM_OF_THREADS; i++) {
      pthread_join(niti[i], NULL);
    }

    // preverimo rezultat
    float rezultat = 0.0;
    for (int i = 0; i<NUM_OF_ELEMENTS; i++) {
        rezultat += vecC[i];
    }
    printf("Rezultat je: %.2f\n", rezultat);

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
  for (int i = 0; i<NUM_OF_ELEMENTS/NUM_OF_THREADS; i++) {
    argumenti->vecStripC[i] = argumenti->vecStripA[i] + argumenti->vecStripB[i];
  }

}