#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int a = 1, b = 1, c = 0;
pthread_barrier_t barrier;

void *prvaNit(void *args) {
    pthread_barrier_wait(&barrier);
    printf("%d + %d = %d\n", a, b, c);
    return NULL;
}

void *drugaNit(void *args) {
    c = a + b;
    pthread_barrier_wait(&barrier);
    return NULL;
}
int main(void) {
    
    pthread_t t1; pthread_t t2;
    pthread_barrier_init(&barrier, NULL, 2);
    pthread_create(&t1, NULL, prvaNit, NULL);
    pthread_create(&t2, NULL, drugaNit, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    
    return 0;

}