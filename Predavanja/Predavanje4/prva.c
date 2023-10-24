#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {

    // ta funkcije vrne koliko nitit se bo ustvarilo
    // num_threads = omp_get_num_teams();
    // printf("Število niti je %d\n", num_threads);

    // vrne max. možno niti
    int max_threads = omp_get_max_threads();
    printf("Max. število niti je: %d\n", max_threads);

    // to pove koliko niti želimo
    omp_set_num_threads(4);
    // to ustvari paralelno sekcijo
    #pragma omp parallel
    { // dejansko moreš tako pisat, sicer je napaka! (oklepaje mislim)
        int num_threads = omp_get_num_teams();
        int myID = omp_get_thread_num(); // <-- ta funkcija vrne ID od niti
        printf("Pozdrav! Jaz sem nit %d od %d niti\n", myID, num_threads);
        // printf("Pozdrav\n");
    } // <-- to predstavlja prepreko, kjer vse niti počakajo.

    return 0;

}