#include <iostream>
#include <vector>

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
using namespace std;
int main() {
    magma_init(); // 初始化MAGMA库
    magma_print_environment();
    cout<<"OK1"<<endl;
    int device;
    magma_queue_t queue;
    magma_getdevice( &device );
    magma_queue_create( device, &queue );
    
    magma_int_t M, N, K;
    magma_int_t *Am, *An, *Bm, *Bn;
    magma_int_t total_size_A_cpu = 0, total_size_B_cpu = 0, total_size_C_cpu = 0;
    magma_int_t total_size_A_dev = 0, total_size_B_dev = 0, total_size_C_dev = 0;
    magma_int_t max_M, max_N, max_K;
    //magma_int_t sizeA, sizeB, sizeC;
    //magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    
    //batchCount is the number of total matrix
    magma_int_t batchCount = 2;

    float *h_A, *h_B, *h_C, *h_Cmagma;
    float *d_A, *d_B, *d_C;
    float *h_A_tmp, *h_B_tmp, *h_C_tmp;
    
    
    float alpha = 1.0;
    float beta  = 0.0;
    float **h_A_array = NULL;
    float **h_B_array = NULL;
    float **h_C_array = NULL;
    float **d_A_array = NULL;
    float **d_B_array = NULL;
    float **d_C_array = NULL;
    
    magma_int_t *h_M, *h_N, *h_K; // hold the sizes on cpu
    magma_int_t *d_M, *d_N, *d_K; // hold the sizes on gpu

    magma_int_t *h_lda, *h_ldda, *d_ldda;
    magma_int_t *h_ldb, *h_lddb, *d_lddb;
    magma_int_t *h_ldc, *h_lddc, *d_lddc;
    
    cout<<"OK2"<<endl;
    // sizes on the cpu
    magma_imalloc_cpu(&h_M, batchCount);
    magma_imalloc_cpu(&h_N, batchCount);
    magma_imalloc_cpu(&h_K, batchCount);
    
    // size arrays on the GPU should be at least of size (batchCount+1)
    magma_imalloc(&d_M, batchCount+1);
    magma_imalloc(&d_N, batchCount+1);
    magma_imalloc(&d_K, batchCount+1);

    // allocate space for the leading dim
    magma_imalloc_cpu(&h_ldda, batchCount);
    magma_imalloc_cpu(&h_lddb, batchCount);
    magma_imalloc_cpu(&h_lddc, batchCount);
    // leading dimension arrays on the GPU should be at least of size (batchCount+1)
    magma_imalloc(&d_ldda, batchCount+1);
    magma_imalloc(&d_lddb, batchCount+1);
    magma_imalloc(&d_lddc, batchCount+1);
    cout<<"OK3"<<endl;
    
    
    cout<<"OK4"<<endl;
    // pointer arrays
    magma_malloc_cpu((void**)&h_A_array, batchCount*sizeof(float*));
    magma_malloc_cpu((void**)&h_B_array, batchCount*sizeof(float*));
    magma_malloc_cpu((void**)&h_C_array, batchCount*sizeof(float*));

    magma_malloc((void**)&d_A_array, batchCount*sizeof(float*));
    magma_malloc((void**)&d_B_array, batchCount*sizeof(float*));
    magma_malloc((void**)&d_C_array, batchCount*sizeof(float*));
    cout<<"OK4.5"<<endl;
    //opts.ntest = 3 (input parameters)
    //opts.niter = 3
    
    //simplify
    
    M = 6;     
    K = 6;
    N = 6;
            //This make h_K is real K but Am and lda transpose in column.
    h_lda = Am = h_M;
    An = h_K;
    h_ldb = Bm = h_K;
    Bn = h_N;
    h_ldc = h_M;

    max_M = max_N = max_K = 0;
    total_size_A_cpu = total_size_B_cpu = total_size_C_cpu = 0;
    total_size_A_dev = total_size_B_dev = total_size_C_dev = 0;

    for (int i = 0; i < batchCount; i++) {
    //This is where he initializes the data
    //Lets try output the random data
    
        h_M[i] = 1 + (rand() % M);
        h_N[i] = 1 + (rand() % N);
        h_K[i] = 1 + (rand() % K);
        max_M = max( max_M, h_M[i] );
        max_N = max( max_N, h_N[i] );
        max_K = max( max_K, h_K[i] );
        cout<<"batchCount "<<i<<" h_M[i] "<<h_M[i]<<" h_N[i] "<<h_N[i]<<" h_K[i] "<<h_K[i]<<endl;
        cout<<"batchCount "<<i<<" max_M "<<max_M<<" max_N "<<max_N<<" max_K "<<max_K<<endl;
        h_ldda[i] =  ((h_lda[i]+31)/32)*32;
        h_lddb[i] =  ((h_ldb[i]+31)/32)*32; 
        h_lddc[i] =  ((h_ldc[i]+31)/32)*32; 

        total_size_A_cpu += An[i] * h_lda[i];
        total_size_A_dev += An[i] * h_ldda[i];

        total_size_B_cpu += Bn[i] * h_ldb[i];
        total_size_B_dev += Bn[i] * h_lddb[i];

        total_size_C_cpu += h_N[i] * h_ldc[i];
        total_size_C_dev += h_N[i] * h_lddc[i];
    }
       
    magma_smalloc_cpu(&h_A,  total_size_A_cpu);
    magma_smalloc_cpu(&h_B,  total_size_B_cpu);
    magma_smalloc_cpu(&h_C,  total_size_C_cpu);
    magma_smalloc_cpu(&h_Cmagma, total_size_C_cpu);

    magma_smalloc(&d_A, total_size_A_dev);
    magma_smalloc(&d_B, total_size_B_dev);
    magma_smalloc(&d_C, total_size_C_dev);
    cout<<"OK5"<<endl;
    cout<<"total_size_A_cpu: "<<total_size_A_cpu<<endl;
    cout<<"total_size_B_cpu: "<<total_size_B_cpu<<endl;
    cout<<"total_size_C_cpu: "<<total_size_C_cpu<<endl;
    /* Initialize the matrices */
    //       h_A[i], h_B[i], h_C[i]
    //clue1:  i+1     i+2      0
    //clue2:  i+1      2       0
    for(int i = 0; i < total_size_A_cpu; ++i){
	h_A[i] = i+1;
    }
    for(int i = 0; i < total_size_B_cpu; ++i){
	h_B[i] = i+1;
    }
    
    for(int i = 0; i < total_size_C_cpu; ++i){
	h_C[i] = 0;
    }
    /* =====================================================================
       Performs operation using MAGMABLAS
       =================================================================== */
    magma_setvector(batchCount, sizeof(magma_int_t), h_M, 1, d_M, 1,queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1,queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_K, 1, d_K, 1,queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_lddb, 1, d_lddb, 1, queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_lddc, 1, d_lddc, 1, queue );
    cout<<"OK6"<<endl;
    h_A_array[0] = d_A;
    h_B_array[0] = d_B;
    h_C_array[0] = d_C;
    cout<<"OK6.1"<<endl;
    for (int i = 1; i < batchCount; i++) {
        h_A_array[i] = h_A_array[i-1] + An[i-1] * h_ldda[i-1];
        h_B_array[i] = h_B_array[i-1] + Bn[i-1] * h_lddb[i-1];
        h_C_array[i] = h_C_array[i-1] + h_N[i-1] * h_lddc[i-1];
    }
    cout<<"OK6.2"<<endl;
    magma_setvector(batchCount, sizeof(float*), h_A_array, 1, d_A_array, 1, queue );
    magma_setvector(batchCount, sizeof(float*), h_B_array, 1, d_B_array, 1, queue );
    magma_setvector(batchCount, sizeof(float*), h_C_array, 1, d_C_array, 1, queue );
    cout<<"OK7"<<endl;
    h_A_tmp = h_A;
    h_B_tmp = h_B;
    h_C_tmp = h_C;
    for (int i = 0; i < batchCount; i++) {
        magma_ssetmatrix( Am[i],  An[i],  h_A_tmp, h_lda[i], h_A_array[i], h_ldda[i], queue );
        magma_ssetmatrix( Bm[i],  Bn[i],  h_B_tmp, h_ldb[i], h_B_array[i], h_lddb[i], queue );
        magma_ssetmatrix( h_M[i], h_N[i], h_C_tmp, h_ldc[i], h_C_array[i], h_lddc[i], queue );
        h_A_tmp += An[i] * h_lda[i];
        h_B_tmp += Bn[i] * h_ldb[i];
        h_C_tmp += h_N[i] * h_ldc[i];
    }
    cout<<"OK8"<<endl;
    //magma_time = magma_sync_wtime(queue );
    magmablas_sgemm_vbatched(MagmaNoTrans,MagmaNoTrans,
                     d_M, d_N, d_K,
                     alpha, d_A_array, d_ldda,
                            d_B_array, d_lddb,
                     beta,  d_C_array, d_lddc,
                     batchCount,
                     queue);
    //magma_time = magma_sync_wtime(queue ) - magma_time;

    cout<<"OK9"<<endl;
    h_C_tmp = h_Cmagma;
    for (int i = 0; i < batchCount; i++) {
        magma_sgetmatrix( h_M[i], h_N[i], h_C_array[i], h_lddc[i], h_C_tmp, h_ldc[i], queue );
        h_C_tmp += h_N[i] * h_ldc[i];
    }
    std::cout << "Matrix C after multiplication:" << std::endl;
    cout<<"OK10"<<endl;
    int offset = 0;
    cout<<"-------------------------------------h_C_tmp:"<<endl;
    for (int i = 0; i < batchCount; ++i) {
       cout<<"Matrix "<<i<<" :"<<endl;
       for(int m = 0; m < h_M[i]; ++m) {
           for(int n = 0; n < h_N[i]; ++n){
               cout << h_C_tmp[offset + n*h_ldc[i] + m] << " ";
           }
       std::cout << std::endl;      
       }
    offset += h_N[i] * h_ldc[i];
    cout<<endl;
    }
    cout<<"-------------------------------------h_Cmagma:"<<endl;
    offset = 0;
    for (int i = 0; i < batchCount; ++i) {
       cout<<"Matrix "<<i<<" :"<<endl;
       for(int m = 0; m < h_M[i]; ++m) {
           for(int n = 0; n < h_N[i]; ++n){
               cout << h_Cmagma[offset + n*h_ldc[i] + m] << " ";
           }
       std::cout << std::endl;      
       }
    offset += h_N[i] * h_ldc[i];
    cout<<endl;
    }
    magma_free_cpu( h_A  );
    magma_free_cpu( h_B  );
    magma_free_cpu( h_C  );
    magma_free_cpu( h_Cmagma  );

    magma_free( d_A );
    magma_free( d_B );
    magma_free( d_C );
  
    // free resources
    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( h_K );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_lddb );
    magma_free_cpu( h_lddc );

    magma_free_cpu( h_A_array  );
    magma_free_cpu( h_B_array  );
    magma_free_cpu( h_C_array  );

    magma_free( d_M );
    magma_free( d_N );
    magma_free( d_K );
    magma_free( d_ldda );
    magma_free( d_lddb );
    magma_free( d_lddc );
    magma_free( d_A_array );
    magma_free( d_B_array );
    magma_free( d_C_array );
    magma_queue_destroy(queue);
    
    magma_finalize();
    return 0;
}


