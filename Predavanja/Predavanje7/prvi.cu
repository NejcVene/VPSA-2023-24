#include <stdio.h>
#include <stdlib.h>
// tole rabimo za CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024

/*
program bo sestavljen iz dveh funkcij, in sicer:
    - funkcija za gostitelja npr. main (prevede se s prevajalnikom za gostitelja)
    - najmanj ene funkcije, ki se izvaja na GPU (prevede se s prevajalnikom za GPU)-
*/

/*
koda znotraj main ne bo videla podatkov, operadnov na napravi (in obratno).
GPU NE MORE vrniti vrednost registru, ki bi ga prebrala CPU.
Skratka ne more vrniti ničesa, zato mora biti void.

Da prevajalnik loči te naše funkcije, moramo tiste za GPU označiti:
    __global__ -> izvaja, prevede na GPU in jo začene CPU
    __local__  -> prevede, izvede in začene na GPU
*/

// koda na napravi (ščepec oz. kernel)
__global__ void sestejVektorje(const float *a, const float *b, float *c, const int n) {
    /*
    a, b, c naslovi posameznik vektorjev
    n -> koliko elementov je v polju

    Zakaj const? Onačuje, da kazalec kaže na nekaj, kar bo shranjeno v pomn. konstant.
    Const ne bo dopuščal spreminjanje vrednosti.

    Potrebno je še ugotoviti, kdo sem in katere elemente lahko seštevam.
    */

    // indeks vektoraja = indeks bloka * velikost + indeks niti v bloku
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tole so vgrajene spremenjivke okolja CUDA in jih lahko beremo kadarkoli med izvajanjem

    if (tid < n) { // če je indeks manjši od dolžine vektorja, smemo delati naslednjo operacijo. Če je večji ne smemo it, saj tam ni nič.
        c[tid] = a[tid] + b[tid];
    }

}

// koda na gostitelju
int main(int argc, char **argv) {

    int dataSize = N * sizeof(float);

    /*
    1. malloc za vektorje a, b in c na gostitelju
    1.1 init vektorjev a in b
    2. malloc za vektorje a, b in c na napravi
    3. kopiraj vektorja a in b na napravo (s uporabo gonilnika, ne load/store ukazi)
    4. zaženi ščepec na napravi
    5. ko se ščepec zaključi, kopiraj rezultat s naprave na gostitelja
    6. preveri in izpiši rezultat
    7. počisti pomnilnik (gostitelja in naprave)
    */

    // drugi teden smo v P18!

    // 1
    float *hVecA = (float *) malloc(dataSize);
    float *hVecB = (float *) malloc(dataSize);
    float *hVecC = (float *) malloc(dataSize);

    // 1.1
    for (int i = 0; i<N i++) {
        hVecA[i] = 1.0;
        hVecB[i] = 2.0;
    }

    // 2
    // Tega me more naredit malloc, zato uporabimo od CUDA
    // uporabi funckijo cudaMalloc((naslov pomnilniške besede na napravi))
    // dobra praksa je, da preberimo kaj nam tale funkcija vrne (če se je uspešno izvedla)
    
    float *dVecA;
    float *dVecB;
    float *dVecC;
    cudaMalloc(&dVecA, dataSize);
    cudaMalloc(&dVecB, dataSize);
    cudaMalloc(&dVecC, dataSize);

    // 3
    // prosi DMA, da te vrednosti prenese
    // uporabi funkcijo cudaMemcpy(void *dst (naprava), vodi *scr (gostitelj), size_t count (koliko bajtov kopirat), cudaMemcpyToDevice (smer prenosa));

    cudaMemcpy(dVecA, hvecA, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dVecB, hvecB, dataSize, cudaMemcpyHostToDevice);

    // 4
    // pokliči funkcijo, ki se zaganja na napravi
    // in pred/ob zagonu definiraj izvajalno okole (koliko blokov v mreži (gridSize), koliko niti v bloku (blokSize))

    // dim3 je struktura
    dim3 blockSize(256, 1, 1); // 256 niti v bloku (1024 je max. število niti v bloku!)
    dim3 gridSize(N / 256, 1, 1); // en blok v mreži
    sestejVektorje<<<gridSize, blockSize>>>(dvecA, dVecB, dVecC, N);

    // 5

    cudaMemcpy(hVecC, dVecC, dataSize, cudaMemcpyDeviceToHost);

    // 6
    float rezultat = 0.0;
    for (int i = 0; i<N; i++) {
        rezultat += hVecA[i];
    }
    printf("Rezultat = %f\n", rezultat); // rezltat mora biti 3072

    // 7

    free(hVecA);
    free(hVecB);
    free(hVecC);

    cudaFree(dVecA);
    cudaFree(dVecB);
    cudaFree(dVecC);

    return 0;

}