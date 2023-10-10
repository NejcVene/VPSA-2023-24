#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // knjižnica, ki ima vse za delo s niti

/*
void* (*neka_funkcija)(int *)
*/

/*
Najprej bomo naredili eno ali več niti s funkcijo
pthread_create()
int pthread_create(pthread_t *restrict thread,
                          const pthread_attr_t *restrict attr,
                          void *(*start_routine)(void *),
                          void *restrict arg);
- pthread_t *restrict thread -> vrne kazalec na ustvarjeno nit. Damo prazen
  kazelec noter, nakar bo ta kazal na ustvarjeno nit, ko se funkcija
  zaključi.
- const pthread_attr_t *restrict attr -> kazalec na množico atributov, ki jih niti ne uporabljamo
- void *(*start_routine)(void *) -> je kazalec na funkcijo, ki vzame
  netipiziran kazalec in vrne netipiziran kazalec. Torej moramo podati
  naslov funckije, ki ustreza temu (da vrne void pointer in vzame void pointer).
  Funkcija create bo, torej na nek način klicala to funkcijo, ki jo mi podamo.
- void *restrict arg je še en netipiziran kazalec, ki je sicer struktura (struct) niti. Tole bomo
  uporabljali za pošiljanje argumentov.
*/

/*
int pthread_join(pthread_t thread, void **retval) 
blokira funkcijo dokler se nit, ki je podana kot argument ne zaključi.
*/

void* funkcija_niti(void *);

// v nit bomo prenašali več argumentov s pomočjo strukture
typedef struct {
  unsigned int thread_id;
} argumenti_t;


pthread_t nit1, nit2;
argumenti_t args1, args2; 

int main(void) {


    args1.thread_id = 1;
    // kreiramo niti
    // edini argument je navaden void naslov (kot to zahteva funkcija), zazo mora naslov na strukturo args1
    // pretvoriti v naslov za navaden void naslov.
    pthread_create(&nit1, NULL, funkcija_niti, (void *) &args1);
    args2.thread_id = 2;
    pthread_create(&nit2, NULL, funkcija_niti, (void *) &args2);

    // počakamo, da se niti zaključita
    pthread_join(nit1, NULL);
    pthread_join(nit2, NULL);

    // sedaj, ko sta se zaključili, lahko končamo main in posledično program.

    return 0;

}

void* funkcija_niti(void *arg) {

  // za dostop do elementov je navaden void naslov, zato moramo
  // definirati nov kazalec na strukturo in ga pretovirit v ta tip.
  argumenti_t *argumenti = (argumenti_t *) arg;
  if (argumenti->thread_id == 1) {
    printf("Sem nit1\n");
  } else if (argumenti->thread_id == 2) {
    printf("Sem nit2\n");
  }

}
