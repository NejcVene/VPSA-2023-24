#include <stdio.h>
#include <stdlib.h>
// tole rabimo za CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#define N (size_t) (1024 * 1024 * 1024) // 1B elementov
#define T (size_t) (1024 * 1024 * 2) // 2M niti

// bs (block size = 128, 256, 512, 1024)
// T = 2^16 (1024 * 64), 2^17, 2^18, 2^19, 2^20
// 20 zagonov skupaj za zgornje parametre
// probi ugotovit, kateri par da najboljše rezultate
// mores tko 3-krat, 4-krat zagnat da dobiš "realen (al kak bi reku)" rezultat

#define BS 1024
#define GS (T / BS)

// size_t = unsigned long (64 bit)

/*
Sedaj imamo problem, ker smo dosegli max. število niti. Kako to odpraviti?
    Ko nit zaključi, se mora njen index predstaviti za toliko, da začne delati
    na drugem bloku. Posledično vsaka nit naredi malo več dela.
*/

// how to run:
// 1. module load CUDA
// 2. nvcc -o program program.cu
// 3. srun --partition=gpu --gpus=1 --ntasks=1 (--mem-per-cpu=4GB tole naj bi povečal kolk spomina mamo na voljo) program

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

// če je možno začenemo toliko niti, kolikor je dolg vektor.
// če je pa več niti kot je dolg vektor, je treba te odvečne niti ustaviti oz. preprečiti
// da ne preberejo spomina, ki ga ne uporabljamo.
// zato smo napisali ta if stavek:
//  if (tid < n) {
//    ...
//  }

// koda na napravi (ščepec oz. kernel)
// __gobal__ pove, da se izvaja na gostitelju (gcc jo bo ignoriral)
__global__ void sestejVektorje(const float *a, const float *b, float *delniSkalarniProdukt, const size_t n) {
    /*
    a, b, c naslovi posameznik vektorjev
    n -> koliko elementov je v polju

    Zakaj const? Onačuje, da kazalec kaže na nekaj, kar bo shranjeno v pomn. konstant.
    Const ne bo dopuščal spreminjanje vrednosti.
BS
    Potrebno je še ugotoviti, kdo sem in katere elemente lahko seštevam.
    */

    // indeks vektoraja = indeks bloka * velikost + indeks niti v bloku
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x; // tole so vgrajene spremenjivke okolja CUDA in jih lahko beremo kadarkoli med izvajanjem

    __shared__ float lokalniDelniProdukti[BS];

    while (tid < n) { // če je indeks manjši od dolžine vektorja, smemo delati naslednjo operacijo. Če je večji ne smemo it, saj tam ni nič.
        [tid] = a[tid] + b[tid]; // vsaka nit sešteje dva istoležna element a v vektorju 
        tid += blockDim.x * gridDim.x; // tole sedaj prestavi nit v drugi blok na isti index
    }

    __syncthreads(); // tale funkcija je prepreka

    if (tid == 0) { // nit s indexom 0
        for (size_t i = 0; i<n; i++) {
            dp += c[i];
        }
    }

}

// koda na gostitelju
// gostitelj je vedno zadolžen za pripravo podatkov, ki se bodo nato uporabili
size_t main(void) {

    size_t dataSize = N * sizeof(float);

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

    // 1
    float *hVecA = (float *) malloc(dataSize);
    float *hVecB = (float *) malloc(dataSize);
    float *hVecC = (float *) malloc(dataSize);

    // 1.1
    for (size_t i = 0; i<N i++) {
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
    size_t bs = 1024; // block size
    dim3 blockSize(bs, 1, 1); // 256 niti v bloku (1024 je max. število niti v bloku!)
    dim3 gridSize(T / bs, 1, 1); // en blok v mreži
    sestejVektorje<<<gridSize, blockSize>>>(dvecA, dVecB, dVecC, N);
    // ^ teti <<<>>> podajo izvajalno okolje

    // 5

    // problematično 
    cudaMemcpy(hVecC, dVecC, dataSize, cudaMemcpyDeviceToHost);

    // 6
    // uporabi double, ker imamo 52 bitov za mantiso in je bolj natančen kot float
    double rezultat = 0.0;
    for (size_t i = 0; i<N; i++) {
        rezultat += (double) hVecA[i];
    }
    prsize_tf("Rezultat = %f\n", rezultat); // rezltat mora biti 3072

    // 7

    free(hVecA);
    free(hVecB);
    free(hVecC);

    cudaFree(dVecA);
    cudaFree(dVecB);
    cudaFree(dVecC);

    return 0;

}

/*
Za merjenje časa:
    Ko se dogodek naredi, si zapomnimo ta čas.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // si znači ta dogodek, torej čas.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    loat miliseconds = 0;
    cudaEventElapsedTime(&milicedonds, start, stop);
    printf("Execution time: %0.3f\n", miliseconds);
*/