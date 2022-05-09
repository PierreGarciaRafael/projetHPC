/* 
 * Sequential implementation of the Block-Lanczos algorithm.
 *
 * This is based on the paper: 
 *     "A modified block Lanczos algorithm with fewer vectors" 
 *
 *  by Emmanuel Thomé, available online at 
 *      https://hal.inria.fr/hal-01293351/document
 *
 * Authors : Charles Bouillaguet
 *
 * v1.00 (2022-01-18)
 * v1.01 (2022-03-13) bugfix with (non-transposed) matrices that have more columns than rows
 *
 * USAGE: 
 *      $ ./lanczos_modp --prime 65537 --n 4 --matrix random_small.mtx
 *
 */
#define _POSIX_C_SOURCE  1  // ctime
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>

#include <mmio.h>

#include <mpi.h>

#include <math.h>

typedef uint64_t u64;
typedef uint32_t u32;

/******************* global variables ********************/

long n = 1;
u64 prime;
char *matrix_filename;
char *kernel_filename;
bool right_kernel = false;
int stop_after = -1;
int blockingSqrt;
long mySize;

int n_iterations;      /* variables of the "verbosity engine" */
double start;
double last_print;
bool ETA_flag;
int expected_iterations;

int MPIsize;
/******************* thread variables ********************/
int MPIrank;
MPI_Comm line_comm;
MPI_Comm col_comm;

/******************* sparse matrix data structure **************/

struct sparsematrix_t {
        int nrows;        // dimensions
        int ncols;
        long int nnz;     // number of non-zero coefficients
        int *i;           // row indices (for COO matrices)
        int *j;           // column indices
        u32 *x;           // coefficients
};

struct block_sparsematrix_t {
        int nrows;        // dimensions
        int ncols;
        int nblocksSqrt;  // number of blocks per rows/columns
        u32 *blockId;     // block end indexes (doesn't contain 0) 
                          // contains number of non-zero coefficients
        
        int *i;           // row indices (for COO matrices)
        int *j;           // column indices
        u32 *x;           // coefficients
};

struct unique_block_t {
        int nrows;        // dimensions
        int ncols;
        int nblocksSqrt;  // number of blocks per rows/columns
        int blockId;      //block coordinate = MPIrank
        int nnz;
        int *i;           // row indices (for COO matrices)
        int *j;           // column indices
        u32 *x;           // coefficients
};

/******************* pseudo-random generator (xoshiro256+) ********************/

/* fixed seed --- this is bad */
u64 rng_state[4] = {0x1415926535, 0x8979323846, 0x2643383279, 0x5028841971};

u64 rotl(u64 x, int k)
{
        u64 foo = x << k;
        u64 bar = x >> (64 - k);
        return foo ^ bar;
}

u64 random64()
{
        u64 result = rotl(rng_state[0] + rng_state[3], 23) + rng_state[0];
        u64 t = rng_state[1] << 17;
        rng_state[2] ^= rng_state[0];
        rng_state[3] ^= rng_state[1];
        rng_state[1] ^= rng_state[2];
        rng_state[0] ^= rng_state[3];
        rng_state[2] ^= t;
        rng_state[3] = rotl(rng_state[3], 45);
        return result;
}

/******************* utility functions ********************/

double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double) ts.tv_sec + ts.tv_usec / 1e6;
}

/* represent n in <= 6 char  */
void human_format(char * target, long n) {
        if (n < 1000) {
                sprintf(target, "%ld", n);
                return;
        }
        if (n < 1000000) {
                sprintf(target, "%.1fK", n / 1e3);
                return;
        }
        if (n < 1000000000) {
                sprintf(target, "%.1fM", n / 1e6);
                return;
        }
        if (n < 1000000000000ll) {
                sprintf(target, "%.1fG", n / 1e9);
                return;
        }
        if (n < 1000000000000000ll) {
                sprintf(target, "%.1fT", n / 1e12);
                return;
        }
}

/************************** command-line options ****************************/

void usage(char ** argv)
{
        printf("%s [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("--matrix FILENAME           MatrixMarket file containing the spasre matrix\n");
        printf("--prime P                   compute modulo P\n");
        printf("--n N                       blocking factor [default 1]\n");
        printf("--block-sqrt N               how many blocks per rows/columns\n");
        printf("--output-file FILENAME      store the block of kernel vectors\n");
        printf("--right                     compute right kernel vectors\n");
        printf("--left                      compute left kernel vectors [default]\n");
        printf("--stop-after N              stop the algorithm after N iterations\n");
        printf("\n");
        printf("The --matrix and --prime arguments are required\n");
        printf("The --stop-after and --output-file arguments mutually exclusive\n");
        exit(0);
}

void process_command_line_options(int argc, char ** argv)
{
        struct option longopts[9] = {
                {"matrix", required_argument, NULL, 'm'},
                {"prime", required_argument, NULL, 'p'},
                {"n", required_argument, NULL, 'n'},
                // {"block-sqrt", required_argument, NULL, 'b'},
                {"output-file", required_argument, NULL, 'o'},
                {"right", no_argument, NULL, 'r'},
                {"left", no_argument, NULL, 'l'},
                {"stop-after", required_argument, NULL, 's'},
                {NULL, 0, NULL, 0}
        };
        char ch;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
                switch (ch) {
                case 'm':
                        matrix_filename = optarg;
                        break;
                case 'n':
                        n = atoi(optarg);
                        break;
                // case 'b':
                //         blockingSqrt = atoi(optarg);
                //         break;
                case 'p':
                        prime = atoll(optarg);
                        break;
                case 'o':
                        kernel_filename = optarg;
                        break;
                case 'r':
                        right_kernel = true;
                        break;
                case 'l':
                        right_kernel = false;
                        break;
                case 's':
                        stop_after = atoll(optarg);
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }

        /* missing required args? */
        if (matrix_filename == NULL || prime == 0)
                usage(argv);
        /* exclusive arguments? */
        if (kernel_filename != NULL && stop_after > 0)
                usage(argv);
        /* range checking */
        if (prime > 0x3fffffdd) {
                errx(1, "p is capped at 2**30 - 35.  Slighly larger values could work, with the\n");
                printf("suitable code modifications.\n");
                exit(1);
        }
}


int matToBlockIndex(int i, int j, int blockWidth, int blockHeight){
        return blockingSqrt * (i/blockHeight) + (j/blockWidth);
}

int blockIndexToStart(int blockIndex, int blockWidth, int blockHeight, int colNb){
        return colNb*blockHeight*blockIndex/blockingSqrt + blockWidth*blockIndex%blockingSqrt;
}

int calcBlockSide(int len, int nBlocks){
        return len/nBlocks + (len%nBlocks > 0);
}

void assertBlockMatrix(struct block_sparsematrix_t * M){
        assert(M->blockId[blockingSqrt*blockingSqrt - 1] == mySize);
        long u = 0;
        int bWidth = calcBlockSide(M->ncols,M->nblocksSqrt);
        int bHeight = calcBlockSide(M->nrows,M->nblocksSqrt);
        for (int blockId = 0; blockId < blockingSqrt*blockingSqrt; blockId += 1){
                for (; u < M->blockId[blockId]; u += 1){ // checking each item is in its block.
                        assert(matToBlockIndex(M->i[u], M->j[u],bWidth,bHeight) == blockId);
                }
        }
        
}

void unique_block_to_sparse_matrix(const struct unique_block_t * B, struct sparsematrix_t *M){
        M->ncols=B->ncols;
        M->nrows=B->nrows;
        M->nnz=B->nnz;
        M->i=B->i;
        M->j=B->j;
        M->x=B->x;
}

/****************** sparse matrix operations ******************/

/* Load a matrix from a file in "blocked list of triplets" representation giving
blocks where it needs to.*/
void block_sparsematrix_mm_load_MPI(struct block_sparsematrix_t * M, char const * filename)
{
        int nrows = 0;
        int ncols = 0;
        long nnz = 0;
        if (MPIrank == 0){
                printf("Loading matrix from %s\n", filename);
                fflush(stdout);
        }

        FILE *f = fopen(filename, "r");
        if (f == NULL)
                err(1, "impossible d'ouvrir %s", filename);

        /* read the header, check format */
        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0)
                errx(1, "Could not process Matrix Market banner.\n");
        if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
                errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", 
                        mm_typecode_to_str(matcode));
        if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                errx(1, "Matrix type [%s] not supported (only integer general are OK)", 
                        mm_typecode_to_str(matcode));
        if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0)
                errx(1, "Cannot read matrix size");
        long myStart = MPIrank*(nnz/MPIsize);
        mySize = (MPIrank == MPIsize-1)? nnz-myStart: nnz/MPIsize;
        fprintf(stderr, "  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, mySize);
        fprintf(stderr, "  - Allocating %.1f MByte, (keeping half of it after load)\n", 1e-6 * (24.0 * mySize));

        int blockWidth = calcBlockSide(ncols,blockingSqrt);
        int blockHeight = calcBlockSide(nrows,blockingSqrt);
        assert(blockingSqrt*blockWidth >= ncols);
        assert(blockingSqrt*blockHeight >= nrows);
        /* Allocate memory for the matrix */
        int *Mi = malloc(mySize * sizeof(*Mi));
        int *Mj = malloc(mySize * sizeof(*Mj));
        u32 *Mx = malloc(mySize * sizeof(*Mx));
        u32 *histogramOfBlocks = malloc(blockingSqrt*blockingSqrt*sizeof(*histogramOfBlocks));

        for (int blockId = 0; blockId < blockingSqrt*blockingSqrt; blockId += 1){
                histogramOfBlocks[blockId] = 0;
        }
        if (Mi == NULL || Mj == NULL || Mx == NULL)
                err(1, "Cannot allocate sparse matrix");

        /* Parse and load actual entries */
        double start = wtime();
        // printf("before line skip, r/s %d/%d, domain [%ld,%ld[, nnz = %ld\n",
        //         MPIrank, MPIsize, myStart, myStart+mySize, nnz);
        int dump = 0;
        for (long toSkip = 0; toSkip < myStart; toSkip ++){  
                fscanf(f, "%d %d %d\n", &dump, &dump, &dump);
        }
        for (long u = 0; u < mySize; u++) {
                int i, j;
                u32 x;
                if (3 != fscanf(f, "%d %d %d\n", &i, &j, &x))
                        errx(1, "parse error entry %ld\n", u);
                Mi[u] = i - 1;  /* MatrixMarket is 1-based */
                Mj[u] = j - 1;
                Mx[u] = x % prime;
                int Bid = matToBlockIndex(i-1, j-1, blockWidth, blockHeight);
                histogramOfBlocks[Bid] +=1;
                
                // verbosity
                if (MPIrank == MPIsize -1 && (u & 0xffff) == 0xffff) {
                        double elapsed = wtime() - start;
                        double percent = (100. * u) / mySize;
                        double rate = ftell(f) / 1048576. / elapsed;
                        printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
                }
        }



        /* finalization */
        fclose(f);
        fflush( stdout );

        

        int *BMi = malloc(mySize * sizeof(*BMi));
        int *BMj = malloc(mySize * sizeof(*BMj));
        u32 *BMx = malloc(mySize * sizeof(*BMx));
        u32 *Bid = malloc(blockingSqrt*blockingSqrt*sizeof(*Bid));
        Bid[0] = 0;
        for (int blockId = 1; blockId < blockingSqrt*blockingSqrt; blockId += 1){
                Bid[blockId] = Bid[blockId-1] + histogramOfBlocks[blockId-1];
        }
        for (long u = 0; u < mySize; u += 1){
                int bIndex = matToBlockIndex(Mi[u],Mj[u],blockWidth,blockHeight);
                u32 index = Bid[bIndex];
                Bid[bIndex] += 1;
                histogramOfBlocks[bIndex] -= 1;
                BMi[index] = Mi[u];
                BMj[index] = Mj[u];
                BMx[index] = Mx[u];
                assert(bIndex < blockingSqrt*blockingSqrt);
                if (bIndex < blockingSqrt*blockingSqrt-1){
                        assert( Bid[bIndex] <= Bid[bIndex + 1]);
                }
        }
        fflush(stdout);
        assert(Bid[blockingSqrt*blockingSqrt-1] == mySize);
        M->nrows = nrows;
        M->ncols = ncols;
        M->blockId = Bid;
        M->nblocksSqrt= blockingSqrt;
        M->i = BMi;
        M->j = BMj;
        M->x = BMx;
        free(Mi);
        free(Mj);
        free(Mx);
        free(histogramOfBlocks);
}

void resizeInt(int ** tab, int previousSize, int newSize, bool initZero){
        
        int * newTab = malloc(newSize*sizeof(int));
        int i;
        if (previousSize > newSize){
                for (i = 0; i < newSize; i += 1){
                        newTab[i] = (*tab)[i];
                }
                free(*tab);
                *tab = newTab;
                return;
        }
        for (i = 0; i < previousSize; i += 1){
                newTab[i] = (*tab)[i];
        }
        if (initZero){
                for (; i < newSize; i += 1){
                        newTab[i] = 0;
                }
        }
        free(*tab);
        *tab = newTab;
}

void resizeU32(u32 ** tab, int previousSize, int newSize, bool initZero){
        printf("%d->%d\n",previousSize, newSize);
        u32 * newTab = malloc(newSize*sizeof(int));
        int i;
        if (previousSize > newSize){
                for (i = 0; i < newSize; i += 1){
                        newTab[i] = (*tab)[i];
                }
                free(*tab);
                *tab = newTab;
                return;
        }
        for (i = 0; i < previousSize; i += 1){
                newTab[i] = (*tab)[i];
        }
        if (initZero){
                for (; i < newSize; i += 1){
                        newTab[i] = 0;
                }
        }
        free(*tab);
        *tab = newTab;
}

/* Load a matrix from a file in "list of triplet" representation */
void load_uniqueblock_sparsematrix(struct unique_block_t * M, char const * filename, int blockIdx)
{
        int nrows = 0;
        int ncols = 0;
        long nnz = 0;

        printf("Loading matrix from %s\n", filename);
        fflush(stdout);

        FILE *f = fopen(filename, "r");
        if (f == NULL)
                err(1, "impossible d'ouvrir %s", filename);

        /* read the header, check format */
        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0)
                errx(1, "Could not process Matrix Market banner.\n");
        if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
                errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", 
                        mm_typecode_to_str(matcode));
        if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                errx(1, "Matrix type [%s] not supported (only integer general are OK)", 
                        mm_typecode_to_str(matcode));
        if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0)
                errx(1, "Cannot read matrix size");
        int blockWidth  = calcBlockSide(ncols,blockingSqrt);
        int blockHeight = calcBlockSide(nrows,blockingSqrt);
        // printf("blockWidth,blockHeight,nnz,ncols,nrows=%d,%d,%ld,%d,%d\n",
        // blockWidth, blockHeight, nnz, ncols, nrows);
        long estimatedSize = 1.1 * nnz/(blockingSqrt*blockingSqrt);
        //'1.1 * ' is to reduce the number of realloc
        long nowSize = 0;
        fprintf(stderr, "  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, nnz);
        fprintf(stderr, "  - Allocating %.1f MByte, it will surely vary\n", 1e-6 * (12.0 * estimatedSize));
        /* Allocate memory for the matrix */
        int *BMi = malloc(estimatedSize * sizeof(*BMi));
        int *BMj = malloc(estimatedSize * sizeof(*BMj));
        u32 *BMx = malloc(estimatedSize * sizeof(*BMx));
        if (BMi == NULL || BMj == NULL || BMx == NULL)
                err(1, "Cannot allocate sparse matrix");

        /* Parse and load actual entries */
        double start = wtime();
        for (long u = 0; u < nnz; u++) {
                int i, j;
                u32 x;
                if (3 != fscanf(f, "%d %d %d\n", &i, &j, &x))
                        errx(1, "parse error entry %ld\n", u);
                int bid = matToBlockIndex(i-1,j-1,blockWidth, blockHeight);
                if(bid==blockIdx){
                        if (nowSize == estimatedSize-1){
                                resizeInt(&BMi, estimatedSize, estimatedSize * 1.2, false);
                                resizeInt(&BMj, estimatedSize, estimatedSize * 1.2, false);
                                resizeU32(&BMx, estimatedSize, estimatedSize * 1.2, false);
                                estimatedSize *= 1.2;
                        }
                        BMi[nowSize] = i-1;
                        BMj[nowSize] = j-1;
                        BMx[nowSize] = x;
                        nowSize += 1;
                }
                // verbosity
                if ((u & 0xffff) == 0xffff) {
                        double elapsed = wtime() - start;
                        double percent = (100. * u) / nnz;
                        double rate = ftell(f) / 1048576. / elapsed;
                        printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
                }
        }
        resizeInt(&BMi, estimatedSize, nowSize, false);
        resizeInt(&BMj, estimatedSize, nowSize, false);
        resizeU32(&BMx, estimatedSize, nowSize, false);
        /* finalization */
        fclose(f);
        printf("\n");
        M->nrows = nrows;
        M->ncols = ncols;
        M->blockId = blockIdx;
        M->i = BMi;
        M->j = BMj;
        M->x = BMx;
        M->nnz = nowSize;
        M->nblocksSqrt = blockingSqrt;
}

/* Load a matrix from a file in "list of triplet" representation */
void sparsematrix_mm_load(struct sparsematrix_t * M, char const * filename)
{
        int nrows = 0;
        int ncols = 0;
        long nnz = 0;

        printf("Loading matrix from %s\n", filename);
        fflush(stdout);

        FILE *f = fopen(filename, "r");
        if (f == NULL)
                err(1, "impossible d'ouvrir %s", filename);

        /* read the header, check format */
        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0)
                errx(1, "Could not process Matrix Market banner.\n");
        if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
                errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", 
                        mm_typecode_to_str(matcode));
        if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                errx(1, "Matrix type [%s] not supported (only integer general are OK)", 
                        mm_typecode_to_str(matcode));
        if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0)
                errx(1, "Cannot read matrix size");
        fprintf(stderr, "  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, nnz);
        fprintf(stderr, "  - Allocating %.1f MByte\n", 1e-6 * (12.0 * nnz));

        /* Allocate memory for the matrix */
        int *Mi = malloc(nnz * sizeof(*Mi));
        int *Mj = malloc(nnz * sizeof(*Mj));
        u32 *Mx = malloc(nnz * sizeof(*Mx));
        if (Mi == NULL || Mj == NULL || Mx == NULL)
                err(1, "Cannot allocate sparse matrix");

        /* Parse and load actual entries */
        double start = wtime();
        for (long u = 0; u < nnz; u++) {
                int i, j;
                u32 x;
                if (3 != fscanf(f, "%d %d %d\n", &i, &j, &x))
                        errx(1, "parse error entry %ld\n", u);
                Mi[u] = i - 1;  /* MatrixMarket is 1-based */
                Mj[u] = j - 1;
                Mx[u] = x % prime;
                
                // verbosity
                if ((u & 0xffff) == 0xffff) {
                        double elapsed = wtime() - start;
                        double percent = (100. * u) / nnz;
                        double rate = ftell(f) / 1048576. / elapsed;
                        printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
                }
        }

        /* finalization */
        fclose(f);
        printf("\n");
        M->nrows = nrows;
        M->ncols = ncols;
        M->nnz = nnz;
        M->i = Mi;
        M->j = Mj;
        M->x = Mx;
}

/* y = M*x or y = transpose(M)*x, according to the transpose flag */ 
void sparse_matrix_vector_product(u32 * y, struct sparsematrix_t const * M, u32 const * x, bool transpose)
{
        long nnz = M->nnz;
        int nrows = transpose ? M->ncols : M->nrows;
        int const * Mi = M->i;
        int const * Mj = M->j;
        u32 const * Mx = M->x;
        
        for (long i = 0; i < nrows * n; i++)
                y[i] = 0;
        printf("nnz = %ld, n = %ld\n",nnz,n);
        for (long k = 0; k < nnz; k++) {
                int i = transpose ? Mj[k] : Mi[k];
                int j = transpose ? Mi[k] : Mj[k];
                u64 v = Mx[k];
                for (int l = 0; l < n; l++) {
                        u64 a = y[i * n + l];
                        u64 b = x[j * n + l];
                        y[i * n + l] = (a + v * b) % prime;
                }
        }
}

/* y = M*x or y = transpose(M)*x, according to the transpose flag */ 
void unique_block_vector_product(u32 *y, struct unique_block_t const * M, u32 const * x, bool transpose)
{
        long nnz = M->nnz;
        int const * Mi = M->i;
        int const * Mj = M->j;
        u32 const * Mx = M->x;
        int nrows = transpose ? M->ncols : M->nrows;
        int ncols = transpose ? M->nrows : M->ncols;
        int startI = (M->blockId/M->nblocksSqrt) * calcBlockSide(nrows, M->nblocksSqrt);
        int startJ = (M->blockId%M->nblocksSqrt) * calcBlockSide(ncols, M->nblocksSqrt);
        for (long i = 0; i < calcBlockSide(ncols,M->nblocksSqrt) * n; i++)
                y[i] = 0;

        for (long k = 0; k < nnz; k++) {
                int i = transpose ? Mj[k] - startJ : Mi[k] - startI;
                int j = transpose ? Mi[k] - startI : Mj[k] - startJ;
                u64 v = Mx[k];
                for (int l = 0; l < n; l++) {
                        u64 a = y[i * n + l];
                        u64 b = x[j * n + l];
                        y[i * n + l] = (a + v * b) % prime;
                }
        }
}

//computes needed tmp for the next sp_mat_vec_prod, by sharing it with other processes.
void getLineCalc(u32 *tmp, long size){
        u32 *toReceive = NULL; //to remove warning : "may be uninitialized"
        if (MPIrank%blockingSqrt == 0)
                toReceive = malloc((blockingSqrt) * (size*sizeof(*toReceive)));
        MPI_Gather(tmp, size, MPI_UINT32_T,
                        toReceive, size, MPI_UINT32_T, 0, line_comm);
        if (MPIrank%blockingSqrt == 0){
                for (int pid=1; pid < blockingSqrt; pid += 1){
                        for(int i = 0; i < size; i += 1){
                                tmp[i] = (tmp[i] + toReceive[pid*size + i])%prime;
                        }
                }
        }
        MPI_Bcast(tmp, size, MPI_UINT32_T, 0, line_comm);
        if (MPIrank%blockingSqrt == 0)
                free(toReceive);
}


/****************** dense linear algebra modulo p *************************/ 

/* C += A*B   for n x n matrices */
void matmul_CpAB(u32 * C, u32 const * A, u32 const * B)
{
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        for (int k = 0; k < n; k++) {
                                u64 x = C[i * n + j];
                                u64 y = A[i * n + k];
                                u64 z = B[k * n + j];
                                C[i * n + j] = (x + y * z) % prime;
                        }
}

/* C += transpose(A)*B   for n x n matrices */
void matmul_CpAtB(u32 * C, u32 const * A, u32 const * B)
{
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        for (int k = 0; k < n; k++) {
                                u64 x = C[i * n + j];
                                u64 y = A[k * n + i];
                                u64 z = B[k * n + j];
                                C[i * n + j] = (x + y * z) % prime;
                        }
}

/* return a^(-1) mod b */
u32 invmod(u32 a, u32 b)
{
        long int t = 0;  
        long int nt = 1;  
        long int r = b;  
        long int nr = a % b;
        while (nr != 0) {
                long int q = r / nr;
                long int tmp = nt;  
                nt = t - q*nt;  
                t = tmp;
                tmp = nr;  
                nr = r - q*nr;  
                r = tmp;
        }
        if (t < 0)
                t += b;
        return (u32) t;
}

/* 
 * Given an n x n matrix U, compute a "partial-inverse" V and a diagonal matrix
 * d such that d*V == V*d == V and d == V*U*d. Return the number of pivots.
 */ 
int semi_inverse(u32 const * M_, u32 * winv, u32 * d)
{
        u32 M[n * n];
        int npiv = 0;
        for (int i = 0; i < n * n; i++)   /* copy M <--- M_ */
                M[i] = M_[i];
        /* phase 1: compute d */
        for (int i = 0; i < n; i++)       /* setup d */
                d[i] = 0;
        for (int j = 0; j < n; j++) {     /* search a pivot on column j */
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i*n + j] != 0) {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue;         /* no pivot found */
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot*n + j], prime);  /* multiply pivot row to make pivot == 1 */
                for (int k = 0; k < n; k++) {
                        u64 x = M[pivot*n + k];
                        M[pivot*n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {   /* swap pivot row with row j */
                        u32 tmp = M[j*n + k];
                        M[j*n + k] = M[pivot*n + k];
                        M[pivot*n + k] = tmp;
                }
                for (int i = 0; i < n; i++) {  /* eliminate everything else on column j */
                        if (i == j)
                                continue;
                        u64 multiplier = M[i*n+j];
                        for (int k = 0; k < n; k++) {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;  
                        }
                }
        }
        /* phase 2: compute d and winv */
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                        M[i*n + j] = (d[i] && d[j]) ? M_[i*n + j] : 0;
                        winv[i*n + j] = ((i == j) && d[i]) ? 1 : 0;
                }
        npiv = 0;
        for (int i = 0; i < n; i++)
                d[i] = 0;
        /* same dance */
        for (int j = 0; j < n; j++) { 
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i*n + j] != 0) {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue;
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot*n + j], prime);
                for (int k = 0; k < n; k++) {
                        u64 x = M[pivot*n + k];
                        M[pivot*n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {
                        u32 tmp = M[j*n + k];
                        M[j*n + k] = M[pivot*n + k];
                        M[pivot*n + k] = tmp;
                }
                for (int k = 0; k < n; k++) {
                        u64 x = winv[pivot * n + k];
                        winv[pivot * n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {
                        u32 tmp = winv[j * n + k];
                        winv[j * n + k] = winv[pivot * n + k];
                        winv[pivot * n + k] = tmp;
                }
                for (int i = 0; i < n; i++) {
                        if (i == j)
                                continue;
                        u64 multiplier = M[i * n + j];
                        for (int k = 0; k < n; k++) {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;
                                u64 w = winv[i * n + k];
                                u64 z = winv[j * n + k];
                                winv[i * n + k] = (w + (prime - multiplier) * z) % prime;  
                        }
                }
        }
        return npiv;
}

/*************************** block-Lanczos algorithm ************************/

/* Computes vtAv <-- transpose(v) * Av, vtAAv <-- transpose(Av) * Av */
void block_dot_products(u32 * vtAv, u32 * vtAAv, int N, u32 const * Av, u32 const * v)
{
        for (int i = 0; i < n * n; i++)
                vtAv[i] = 0;
        for (int i = 0; i < N; i += n)
                matmul_CpAtB(vtAv, &v[i*n], &Av[i*n]);
        
        for (int i = 0; i < n * n; i++)
                vtAAv[i] = 0;
        for (int i = 0; i < N; i += n)
                matmul_CpAtB(vtAAv, &Av[i*n], &Av[i*n]);
}

/* Compute the next values of v (in tmp) and p */
void orthogonalize(u32 * v, u32 * tmp, u32 * p, u32 * d, u32 const * vtAv, const u32 *vtAAv, 
        u32 const * winv, int N, u32 const * Av)
{
        /* compute the n x n matrix c */
        u32 c[n * n];
        u32 spliced[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                        spliced[i*n + j] = d[j] ? vtAAv[i * n + j] : vtAv[i * n + j];
                        c[i * n + j] = 0;
                }
        matmul_CpAB(c, winv, spliced);
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        c[i * n + j] = prime - c[i * n + j];

        u32 vtAvd[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        vtAvd[i*n + j] = d[j] ? prime - vtAv[i * n + j] : 0;

        /* compute the next value of v ; store it in tmp */        
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        tmp[i*n + j] = d[j] ? Av[i*n + j] : v[i * n + j];
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i*n], &v[i*n], c);
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i*n], &p[i*n], vtAvd);
        
        /* compute the next value of p */
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        p[i * n + j] = d[j] ? 0 : p[i * n + j];
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&p[i*n], &v[i*n], winv);
}

void verbosity()
{
        n_iterations += 1;
        double elapsed = wtime() - start;
        if (elapsed - last_print < 1) 
                return;

        last_print = elapsed;
        double per_iteration = elapsed / n_iterations;
        double estimated_length = expected_iterations * per_iteration;
        time_t end = start + estimated_length;
        if (!ETA_flag) {
                int d = estimated_length / 86400;
                estimated_length -= d * 86400;
                int h = estimated_length / 3600;
                estimated_length -= h * 3600;
                int m = estimated_length / 60;
                estimated_length -= m * 60;
                int s = estimated_length;
                printf("    - Expected duration : ");
                if (d > 0)
                        printf("%d j ", d);
                if (h > 0)
                        printf("%d h ", h);
                if (m > 0)
                        printf("%d min ", m);
                printf("%d s\n", s);
                ETA_flag = true;
        }
        char ETA[30];
        ctime_r(&end, ETA);
        ETA[strlen(ETA) - 1] = 0;  // élimine le \n final
        printf("\r    - iteration %d / %d. %.3fs per iteration. ETA: %s", 
                n_iterations, expected_iterations, per_iteration, ETA);
        fflush(stdout);
}

/* optional tests */
void correctness_tests(u32 const * vtAv, u32 const * vtAAv, u32 const * winv, u32 const * d)
{
        /* vtAv, vtAAv, winv are actually symmetric + winv and d match */
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++) {
                        assert(vtAv[i*n + j] == vtAv[j*n + i]);
                        assert(vtAAv[i*n + j] == vtAAv[j*n + i]);
                        assert(winv[i*n + j] == winv[j*n + i]);
                        assert((winv[i*n + j] == 0) || d[i] || d[j]);
                }
        /* winv satisfies d == winv * vtAv*d */
        u32 vtAvd[n * n];
        u32 check[n * n];
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++) {
                        vtAvd[i*n + j] = d[j] ? vtAv[i*n + j] : 0;
                        check[i*n + j] = 0;
                }
        matmul_CpAB(check, winv, vtAvd);
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++)
                        if (i == j)
                                assert(check[j*n + j] == d[i]);
                        else
                                assert(check[i*n + j] == 0);
}

/* check that we actually computed a kernel vector */
void final_check(int nrows, int ncols, u32 const * v, u32 const * vtM)
{
        printf("Final check:\n");
        /* Check if v != 0 */
        bool good = false;
        for (long i = 0; i < nrows; i++)
                for (long j = 0; j < n; j++)
                        good |= (v[i*n + j] != 0);
        if (good)
                printf("  - OK:    v != 0\n");
        else
                printf("  - KO:    v == 0\n");
                
        /* tmp == Mt * v. Check if tmp == 0 */
        good = true;
        for (long i = 0; i < ncols; i++)
                for (long j = 0; j < n; j++)
                        good &= (vtM[i*n + j] == 0);
        if (good)
                printf("  - OK: vt*M == 0\n");
        else
                printf("  - KO: vt*M != 0\n");                
}

/**************************** dense vector block IO ************************/

void save_vector_block(char const * filename, int nrows, int ncols, u32 const * v)
{
        printf("Saving result in %s\n", filename);
        FILE * f = fopen(filename, "w");
        if (f == NULL)
                err(1, "cannot open %s", filename);
        fprintf(f, "%%%%MatrixMarket matrix array integer general\n");
        fprintf(f, "%%block of left-kernel vector computed by lanczos_modp\n");
        fprintf(f, "%d %d\n", nrows, ncols);
        for (long j = 0; j < ncols; j++)
                for (long i = 0; i < nrows; i++)
                        fprintf(f, "%d\n", v[i*n + j]);
        fclose(f);
}


void save_sparse_matrix(char const * filename, struct sparsematrix_t const * M)
{
        printf("Saving result in %s\n", filename);
        FILE * f = fopen(filename, "w");
        if (f == NULL)
                err(1, "cannot open %s", filename);
        fprintf(f, "%%%%MatrixMarket spmm integer general\n");
        fprintf(f, "%d %d %ld\n", M->nrows, M->ncols, M->nnz);
        for (long u = 0; u < M->nnz; u++)
                fprintf(f, "%d %d %d\n", M->i[u]+1, M->j[u]+1, M->x[u]);
        fclose(f);
}


/* Solve x*M == 0 or M*x == 0 (if transpose == True) */
u32 * unique_block_lanczos(struct unique_block_t const * M, int n, bool transpose)
{
        printf("Block Lanczos parallel r/s %d/%d\n", MPIrank, MPIsize);

        /************* preparations **************/

        /* allocate blocks of vectors */
        int nrows = transpose ? M->ncols : M->nrows;
        int ncols = transpose ? M->nrows : M->ncols;
        long block_size = calcBlockSide(nrows, blockingSqrt) * n;
        long Npad = ((calcBlockSide(nrows, blockingSqrt) + n - 1) / n) * n;
        long Mpad = ((calcBlockSide(ncols, blockingSqrt) + n - 1) / n) * n;
        long block_size_pad = (Npad > Mpad ? Npad : Mpad) * n;
        char human_size[8];
        human_format(human_size, 4 * sizeof(int) * block_size_pad);
        printf("  - Extra storage needed: %sB\n", human_size);
        u32 *v = malloc(sizeof(*v) * block_size_pad);
        u32 *tmp = malloc(sizeof(*tmp) * block_size_pad);
        u32 *Av = malloc(sizeof(*Av) * block_size_pad);
        u32 *p = malloc(sizeof(*p) * block_size_pad);
        if (v == NULL || tmp == NULL || Av == NULL || p == NULL)
                errx(1, "impossible d'allouer les blocs de vecteur");
        
        /* warn the user */
        expected_iterations = 1 + ncols / n;
        char human_its[8];
        human_format(human_its, expected_iterations);
        printf("  - Expecting %s iterations\n", human_its);
        
        /* prepare initial values */
        for (long i = 0; i < block_size_pad; i++) {
                Av[i] = 0;
                v[i] = 0;
                p[i] = 0;
                tmp[i] = 0;
        }
        //in order to have the same start as single proc lanczos
        if (MPIrank%blockingSqrt == 0){
                for (int i = 0; i < MPIrank/blockingSqrt; i += 1)
                        for (long i = 0; i < block_size; i++)
                                v[0] = random64() % prime;
        }
        MPI_Bcast(rng_state,4,MPI_UINT64_T,0,line_comm);
        //we do all of this
        for (long i = 0; i < block_size; i++)
                v[i] = random64() % prime;


        char name[30];
        sprintf(name,"startVec%d.mtx",MPIrank);
        printf("saving %s\n", name);
        save_vector_block(name,
        block_size/n, n, v);
        

        /************* main loop *************/
        printf("  - Main loop\n");
        start = wtime();
        bool stop = false;
        while (true) {
                if (stop_after > 0 && n_iterations == stop_after)
                        break;
                unique_block_vector_product(tmp, M, v, !transpose);
                getLineCalc(tmp, block_size);
                if (MPIrank % blockingSqrt == 0){
                        char name[30];
                        sprintf(name,"test_parallel%d.mtx",MPIrank/blockingSqrt);
                        printf("saving %s\n", name);
                        save_vector_block(name,
                        calcBlockSide(nrows, blockingSqrt), n, tmp);
                }
                unique_block_vector_product(Av, M, tmp, transpose);
                getLineCalc(Av, block_size);
                break;
                u32 vtAv[n * n];
                u32 vtAAv[n * n];
                block_dot_products(vtAv, vtAAv, nrows, Av, v);

                u32 winv[n * n];
                u32 d[n];
                stop = (semi_inverse(vtAv, winv, d) == 0);

                /* check that everything is working ; disable in production */
                correctness_tests(vtAv, vtAAv, winv, d);
                        
                if (stop)
                        break;

                orthogonalize(v, tmp, p, d, vtAv, vtAAv, winv, nrows, Av);

                /* the next value of v is in tmp ; copy */
                for (long i = 0; i < block_size; i++)
                        v[i] = tmp[i];

                verbosity();
        }
        printf("\n");

        if (stop_after < 0)
                final_check(nrows, ncols, v, tmp);
        printf("  - Terminated in %.1fs after %d iterations, r/s=%d/%d\n", wtime() - start, n_iterations, MPIrank, MPIsize);
        free(tmp);
        free(Av);
        free(p);
        return v;
}

/* Solve x*M == 0 or M*x == 0 (if transpose == True) */
u32 * block_lanczos(struct sparsematrix_t const * M, int n, bool transpose)
{
        printf("Block Lanczos\n");

        /************* preparations **************/

        /* allocate blocks of vectors */
        int nrows = transpose ? M->ncols : M->nrows;
        int ncols = transpose ? M->nrows : M->ncols;
        long block_size = nrows * n;
        long Npad = ((nrows + n - 1) / n) * n;
        long Mpad = ((ncols + n - 1) / n) * n;
        long block_size_pad = (Npad > Mpad ? Npad : Mpad) * n;
        char human_size[8];
        human_format(human_size, 4 * sizeof(int) * block_size_pad);
        printf("  - Extra storage needed: %sB\n", human_size);
        u32 *v = malloc(sizeof(*v) * block_size_pad);
        u32 *tmp = malloc(sizeof(*tmp) * block_size_pad);
        u32 *Av = malloc(sizeof(*Av) * block_size_pad);
        u32 *p = malloc(sizeof(*p) * block_size_pad);
        if (v == NULL || tmp == NULL || Av == NULL || p == NULL)
                errx(1, "impossible d'allouer les blocs de vecteur");
        
        /* warn the user */
        expected_iterations = 1 + ncols / n;
        char human_its[8];
        human_format(human_its, expected_iterations);
        printf("  - Expecting %s iterations\n", human_its);
        
        /* prepare initial values */
        for (long i = 0; i < block_size_pad; i++) {
                Av[i] = 0;
                v[i] = 0;
                p[i] = 0;
                tmp[i] = 0;
        }

        for (long i = 0; i < block_size; i++)
                v[i] = random64() % prime;
        char name[30];
        sprintf(name,"startVecSingle.mtx");
        printf("saving %s\n", name);
        save_vector_block(name,
        block_size/n, n, v);

        /************* main loop *************/
        printf("  - Main loop\n");
        start = wtime();
        bool stop = false;
        while (true) {
                if (stop_after > 0 && n_iterations == stop_after)
                        break;

                sparse_matrix_vector_product(tmp, M, v, !transpose);
                save_vector_block("test_single.mtx",nrows, n, tmp);
                sparse_matrix_vector_product(Av, M, tmp, transpose);
                break;
                u32 vtAv[n * n];
                u32 vtAAv[n * n];
                block_dot_products(vtAv, vtAAv, nrows, Av, v);

                u32 winv[n * n];
                u32 d[n];
                stop = (semi_inverse(vtAv, winv, d) == 0);

                /* check that everything is working ; disable in production */
                correctness_tests(vtAv, vtAAv, winv, d);
                        
                if (stop)
                        break;

                orthogonalize(v, tmp, p, d, vtAv, vtAAv, winv, nrows, Av);

                /* the next value of v is in tmp ; copy */
                for (long i = 0; i < block_size; i++)
                        v[i] = tmp[i];

                verbosity();
        }
        printf("\n");

        if (stop_after < 0)
                final_check(nrows, ncols, v, tmp);
        printf("  - Terminated in %.1fs after %d iterations\n", wtime() - start, n_iterations);
        free(tmp);
        free(Av);
        free(p);
        return v;
}


void singleProc(int argc, char **argv){
        printf("doing single proc\n");
        process_command_line_options(argc, argv);
        struct sparsematrix_t M;

        sparsematrix_mm_load(&M, matrix_filename);

        u32 *kernel = block_lanczos(&M, n, right_kernel);
 
        if (kernel_filename)
                save_vector_block(kernel_filename, right_kernel ? M.ncols : M.nrows, n, kernel);
        else
                printf("Not saving result (no --output given)\n");
        free(kernel);
        
        printf("finished single proc\n");
}

/*************************** main function *********************************/

int main(int argc, char ** argv)
{
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &MPIrank);
        MPI_Comm_size(MPI_COMM_WORLD, &MPIsize);
        if (MPIrank == 1){
                singleProc(argc,argv);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        blockingSqrt = sqrtl(MPIsize);
        MPI_Comm_split(MPI_COMM_WORLD, MPIrank/blockingSqrt, MPIrank, &line_comm);
        MPI_Comm_split(MPI_COMM_WORLD, MPIrank%blockingSqrt, MPIrank, &col_comm);
        process_command_line_options(argc, argv);
        
        
        struct unique_block_t M;
        
        load_uniqueblock_sparsematrix(&M, matrix_filename, MPIrank);
        // printf("finished load, r/s : %d/%d\n", MPIrank, MPIsize);
        // printf("saving what I read\n");
        // struct sparsematrix_t T;
        // unique_block_to_sparse_matrix(&M, &T);
        // char name[20];
        // sprintf(name, "blockMtx%d.mtx", MPIrank);
        // save_sparse_matrix(name,&T);
        
        u32 *kernel = unique_block_lanczos(&M, n, right_kernel);
 
        if (kernel_filename)
                save_vector_block(kernel_filename, right_kernel ? M.ncols : M.nrows, n, kernel);
        else
                printf("Not saving result (no --output given)\n");
        free(kernel);
        
        MPI_Finalize();
        exit(EXIT_SUCCESS);
        
}