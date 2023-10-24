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
    // ker smo dali "for" v pragma, ni potrebno pisat {}, ker sam
    // zanko paralelizira
    #pragma omp parallel for
        for (int i = 0; i<8; i++) {
            printf("Sem nit %d in izvajam interaijo %d\n", 
                    omp_get_thread_num(), 
                    i);
        }

    

    return 0;

}