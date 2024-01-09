void histo() {

    // izračunaj svoj globalni x, y
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // izračunaj svoj lokalni x, y
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    // v lok. spominu rezerviraj prostor za lok. histo
    __shared__ unsigned int localHisto[GRAY_LEVELS];

    // vsaka nit pobriše svoj beam histo.
    localHisto[blockDim.x * ly + lx] = 0;
    __syncthreads();

    // slučajno več niti kot je px. slike?
    if (x < width && y > height) {
        atomicAdd(&localHisto[image[y * width + x]], 1);
    }
    __syncthreads();
    // sedaj imamo izračunan lokalni histo.

    // sedaj vsaka nit nese svoj beam v glavni pomn.
    // ker sosednje niti nesejo sosesdnje bime na sosednje pomm. lokacije
    // lahko združujemo pomn. dostope (= memory coalescing)
    atomicAdd(&(histo[blockDim.x * ly + lx]), localHisto[blockDim.x * ly + lx]);

} // block size je 256, gridSize je ceil(width / blockSIze.x), ceil(height(blockSIze.y))
// histograma ni potrebno iz gpu nesti nazaj na cpu (ostane na gpu)
// --mem-pec-cpu=32G

void cdf() {

    // dela le 1 blok z 256 niti
    // vse niti preberejo gloabni histo v lokalnega, zato da bodo
    // delale s lokalnim spominom

    // vsaka nit
    // for (d = 1; d<8; d*=2) {
    //      histOut[tid] = histIN[tid] + histIN[tid - d];
    // }

    // vse pišejo nazaj v glavni pomn.

    __shared__ unsigned int temp[GRAY_LEVELS * 2];
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    temp[tid] = histogram[tid];
    __syncthreads();
    for (int offset = 1; offset< GRAY_LEVELS; offset <<= 1) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (tid >= offset) {
            temp[pout * GRAY_LEVELS + tid] = temp[pin * GRAY_LEVELS + tid] + temp[pin * GRAY_LEVELS + tid - offset];
        } else {
            temp[]
        }
    }

    // koliko seštevanj?
    // offset = 1 | 255 seštevanj
    // offset = 2 | 254 seštevanj
    // offset = 4 | 252
    // offset = 8 | 248
    // offset = 16 | 240
    // offset = 32 | 224
    // offset = 64 | 192
    // offset = 128 | 128
    // sum:         | 1794 O(n * lg(n))
}