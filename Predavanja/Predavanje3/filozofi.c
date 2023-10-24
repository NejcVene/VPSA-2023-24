#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

/*
tale program je allo over the place, zato ta najpomembejši del
ni prav. Čisto zadnji komentar v psevdokodi prikazuje kako
prav narediti. Uporablj tudi neke zakastnitve (usleep), za katere
ne vem kaj je njihov pomen.
*/

// niti so filozofi
#define NUM_OF_THREADS 5

void* funkcija_niti(void *);

typedef struct {
    unsigned int thread_id; 
} argumenti_t;

pthread_t niti[NUM_OF_THREADS];
argumenti_t argumenti[NUM_OF_THREADS];
pthread_mutex_t kljucavnica[NUM_OF_THREADS];

int main(void) {  

    for (int i = 0; i<NUM_OF_THREADS; i++) {
      argumenti[i].thread_id = i;
      pthread_create(&niti[i], NULL, funkcija_niti, (void *) &argumenti[i]);
    }

    for (int i = 0; i<NUM_OF_THREADS; i++) {
      pthread_join(niti[i], NULL);
    }

    for (int i = 0; i<NUM_OF_THREADS; i++) {
      pthread_mutex_destroy(&kljucavnica[i]);
    }

    return 0;

}

void* funkcija_niti(void *arg) {

  argumenti_t *argumenti = (argumenti_t *) arg;
  for (int i = 0; i<4; i++) {
    while () {
      pthread_mutex_lock(&kljucavnica[(argumenti->thread_id + 1) % NUM_OF_THREADS]);
      if (phtread_mutex_trylock(&kljucavnica[(argumenti->thread_id + 1) % NUM_OF_THREADS]) != SUCCESS) {
        pthread_mutex_unlock(&kljucavnica[(argumenti->thread_id)]);
      } else {
        break;
      }

      }
    }
    // pograbi levo palco (zakleni svojo levo ključavnico)
    pthread_mutex_lock(&kljucavnica[(argumenti->thread_id + 1) % NUM_OF_THREADS]);
    printf("Sem nit %d in pograbim levo palco %d\n", argumenti->thread_id, argumenti->thread_id + 1);
    // pograbi desno paclo (zakleni svojo desno ključavnico)
    pthread_mutex_lock(&kljucavnica[argumenti->thread_id]);
    printf("Sem nit %d in pograbim desno palco %d\n", argumenti->thread_id, argumenti->thread_id);
    // jej
    // odloži palci
    pthread_mutex_unlock(&kljucavnica[argumenti->thread_id]);
    pthread_mutex_unlock(&kljucavnica[(argumenti->thread_id + 1) % NUM_OF_THREADS]);
    // razmišljaj

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

/*
tolo zanko damo znotraj for zanke, ki jo že imamo.
while (1) {
  pthread_mutex_lock(&palcka[mojID]);
  if (try_lock(leva) == SUCCESS) {
    break;
  }
  pthread_mutex_unlock(&palcka[mojID]);
}
Če ne bi bilo try_lock bi se naredi dead lock (smrti objem)
*/