#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_OF_ELEMENTS 1024 * 1024
#define NUM_OF_THREADS 8

void* funkcija_niti(void *);

typedef struct {
    unsigned int thread_id;
    float *vecStripA, *vecStripB, *vecStripC, *localSum;  
} argumenti_t;

pthread_t niti[NUM_OF_THREADS];
argumenti_t argumenti[NUM_OF_THREADS];
float rezultat = 0.0;
// array za hranjenje delnih vsot od niti
float delneVsote[NUM_OF_THREADS];

// tale more bit obvezno globalna
pthread_mutex_t kljucavnica;

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
      argumenti[i].localSum = &delneVsote[i];
      pthread_create(&niti[i], NULL, funkcija_niti, (void *) &argumenti[i]);
    }

    for (int i = 0; i<NUM_OF_THREADS; i++) {
      pthread_join(niti[i], NULL);
    }

    for (int i = 0; i<NUM_OF_THREADS; i++) {
      rezultat += delneVsote[i];
    }

    // sedaj se še rezultat hitreje preveri
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
    // seštejemo vektorja (seštevamo istoležne elemente, ker tako dela seštevanje vektorjev)
    argumenti->vecStripC[i] = argumenti->vecStripA[i] + argumenti->vecStripB[i];
    *(argumenti->localSum) += argumenti->vecStripC[i]; // seštej delne rezultate
  }

}

/*
Zdej mamo tukaj problem, da je rezultat (ko preverimo, če je pravilen) vedno drugačen.
Uporabiti moramo zaklepanje pomnilnika, ker si trenutno niti prepisujejo drugo čez drugo.
*/
/*
Določen del je treba izvedit atomično, in sicer tako ima CPE podporo za to:
  npr. en poseben signal BLOCK.
  lahko dodamo tudi poseben ukaz, ki omogoča, da se read-modify-write ukaz izvede v 
  eni urini periodi - en tak ukaz je xchg.
  še tak ukazi so npr. ll (load link)
*/

/*
Za zaklepanje mamo več funkcij. Trenutno bomo uporabili
pthread_mutex_lock in pthread_mutex_unlock.
*/

/*
Ključavnic se je potrebno pri parelelnem programiranju izogibat!
*/