#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 4
#define ROW 3
#define COLUMN 4

float *pArray;

float **pRows;

int main(void) {

    /*
    apperently, tole ni vredu, ker
    je pRows statično,
    druga dela dva sta pa na kopici.
    Hujše je pa še, da ni zagotovila, da bosta teta dva seznama
    eden za drugem v pomn. Torej ni lokalnosti.
    Če imaš sezname enako dolge, uporabi drugi način (tist ki ni **)
    Če maš pa sezname drugačnih velikosti pa uporabi prvi način (tist ki ima **)
    */
    pRows = (float**) malloc(ROW * sizeof(float *));
    for (int i = 0; i<ROW; i++) {
        pRows[i] = (float *) malloc(COLUMN * sizeof(float));
    }

    for (int i = 0; i<COLUMN; i++) {
        free(pRows[i]);
    }
    free(pRows);

    // to je 2D tabela
    pArray = (float*) (malloc(ROW * COLUMN * sizeof(float)));
    for (int row = 0; row<ROW; row++) {
        for (int column = 0; column<COLUMN; column++) {
            *(pArray + row * COLUMN + column) = (float) row;
            // ^prvi odmik | ^drugi odmik
            // row * col + col
        }
    }
    free(pArray);

    return 0;

}