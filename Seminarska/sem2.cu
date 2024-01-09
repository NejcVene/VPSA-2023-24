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