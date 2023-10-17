#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // knjižnica, ki ima vse za delo s niti

// https://man7.org/linux/man-pages/man3/pthread_create.3.htmls

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
- void *restrict arg je še en netipiziran kazalec, ki je sicer struktura (struct) niti
*/

/*
int pthread_join(pthread_t thread, void **retval) 
blokira funkcijo dokler se nit, ki je podana kot argument ne zaključi.
*/

void* funkcija_niti1(void *);
void* funkcija_niti2(void *);


pthread_t nit1, nit2; 

int main(void) {

    // kreiramo niti
    pthread_create(&nit1, NULL, funkcija_niti1, NULL);
    pthread_create(&nit2, NULL, funkcija_niti2, NULL);

    // počakamo, da se niti zaključita
    pthread_join(nit1, NULL);
    pthread_join(nit2, NULL);

    // sedaj, ko sta se zaključili, lahko končamo main in posledično program.

    return 0;

}

void* funkcija_niti1(void* arg) {

    printf("Sem nit1\n");

}

void* funkcija_niti2(void* arg) {

    printf("Sem nit2\n");

}