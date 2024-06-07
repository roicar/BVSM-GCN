#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <filesystem>
#include <algorithm>
// includes, project
#include "magma_v2.h"

using namespace std;
namespace fs = std::filesystem;
#define N_RUNS 10


void verify(float* matrix1,float* matrix2,int batchCount){
    for(int i=0;i<batchCount;i++){
        if(abs(matrix1[i]-matrix2[i])>1.0e-2){
            cout << "The result of matrix multiplication is wrong!" << endl;
            return;
        }

    }
    cout <<  "The result of matrix multiplication is true!" << endl;
}
//获取特定开头的文件名
vector<string> getFileNames(const string& path, const string& prefix) {
    vector<string> fileNames;
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            string fileName = entry.path().filename().string();
            if (fileName.substr(0, prefix.length()) == prefix) {
                fileNames.push_back(fileName);
            }
        }
    }
    return fileNames;
}


//读取文件到指定容器内
vector<vector<float>> readFile(const string& filename) {
    ifstream file(filename);
    vector <vector<float>> data;

    if (file) {
        string line;
        while (std::getline(file, line)) {
            vector <float> row;
            stringstream lineStream(line);
            string cell;

            while (getline(lineStream, cell, ',')) {
                float cellf = stof(cell);
                row.push_back(cellf);
            }

            data.push_back(row);
        }

        file.close();
    } else {
        cout << "Failed to open file: " << filename << endl;
    }
    return data;
}

void compute(vector<vector<vector<vector<float>>>> matrix_A_total,vector<vector<vector<vector<float>>>> matrix_B_total,vector<vector<vector<vector<float>>>> matrix_C_total, int MMnumber, int M_max, int K_max, int N_max) {
    
    
    magma_init(); // 初始化MAGMA库
    magma_print_environment();
    //cout<<"OK1"<<endl;
    int len = matrix_A_total.size();
    cout<<"len: "<<len<<endl;
    int device;
    magma_queue_t queue;
    magma_getdevice( &device );
    magma_queue_create( device, &queue );
    
    //magma_int_t M, N, K;
    magma_int_t *Am, *An, *Bm, *Bn;
    magma_int_t total_size_A_cpu = 0, total_size_B_cpu = 0, total_size_C_cpu = 0;
    magma_int_t total_size_A_dev = 0, total_size_B_dev = 0, total_size_C_dev = 0;

    magma_int_t batchCount = MMnumber;

    float *h_A, *h_B, *h_C, *h_Cmagma, *c_rel;
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
    
    
    //magma_timer_t time,time_sum=0;
    //cout<<"OK2"<<endl;
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
    //cout<<"OK3"<<endl;
    
    
    //cout<<"OK4"<<endl;
    // pointer arrays
    magma_malloc_cpu((void**)&h_A_array, batchCount*sizeof(float*));
    magma_malloc_cpu((void**)&h_B_array, batchCount*sizeof(float*));
    magma_malloc_cpu((void**)&h_C_array, batchCount*sizeof(float*));

    magma_malloc((void**)&d_A_array, batchCount*sizeof(float*));
    magma_malloc((void**)&d_B_array, batchCount*sizeof(float*));
    magma_malloc((void**)&d_C_array, batchCount*sizeof(float*));
    //cout<<"OK4.5"<<endl;

    h_lda = Am = h_M;
    An = h_K;
    h_ldb = Bm = h_K;
    Bn = h_N;
    h_ldc = h_M;

    total_size_A_cpu = total_size_B_cpu = total_size_C_cpu = 0;
    total_size_A_dev = total_size_B_dev = total_size_C_dev = 0;
    
    int offsetin = 0;
    int batchin;
    for (int i = 0; i < len; i++) {
        batchin = matrix_A_total[i].size();
        for(int j = 0; j < batchin; j++){
		h_M[offsetin +j] = matrix_A_total[i][0].size();
		h_N[offsetin +j] = matrix_B_total[i][0][0].size();
		h_K[offsetin +j] = matrix_A_total[i][0][0].size();

		h_ldda[offsetin +j] =  ((h_lda[offsetin +j]+31)/32)*32;
		h_lddb[offsetin +j] =  ((h_ldb[offsetin +j]+31)/32)*32; 
		h_lddc[offsetin +j] =  ((h_ldc[offsetin +j]+31)/32)*32; 

		total_size_A_cpu += An[offsetin +j] * h_lda[offsetin +j];
		total_size_A_dev += An[offsetin +j] * h_ldda[offsetin +j];

		total_size_B_cpu += Bn[offsetin +j] * h_ldb[offsetin +j];
		total_size_B_dev += Bn[offsetin +j] * h_lddb[offsetin +j];

		total_size_C_cpu += h_N[offsetin +j] * h_ldc[offsetin +j] ;
		total_size_C_dev += h_N[offsetin +j] * h_lddc[offsetin +j];
        }
        offsetin += batchin;
    }
    //cout<<"OK4.9"<<endl;   
    magma_smalloc_cpu(&h_A,  total_size_A_cpu);
    magma_smalloc_cpu(&h_B,  total_size_B_cpu);
    magma_smalloc_cpu(&h_C,  total_size_C_cpu);
    magma_smalloc_cpu(&h_Cmagma, total_size_C_cpu);
    magma_smalloc_cpu(&c_rel, total_size_C_cpu);

    magma_smalloc(&d_A, total_size_A_dev);
    magma_smalloc(&d_B, total_size_B_dev);
    magma_smalloc(&d_C, total_size_C_dev);
    //cout<<"OK5"<<endl;
    //cout<<"total_size_A_cpu: "<<total_size_A_cpu<<endl;
    //cout<<"total_size_B_cpu: "<<total_size_B_cpu<<endl;
    //cout<<"total_size_C_cpu: "<<total_size_C_cpu<<endl;
    int offset = 0;
    int m,k,n,batch;
    m = k = n = batch = 0;
    for(int p = 0; p < len; p++){
        m = matrix_A_total[p][0].size();
        k = m;
        batch = matrix_A_total[p].size();
        cout<<"m: "<<m<<endl;
        cout<<"k: "<<k<<endl;
        cout<<"batch: "<<batch<<endl;
        for(int i=0; i<batch; i++) {
		for(int t=0; t<k; t++){
		    for(int j=0; j<m; j++){
			    //cout<<matrix_A_total[p][i][j][t]<<" ";
			    h_A[offset + i*m*k + t*m +j] = matrix_A_total[p][i][j][t];			
			}
		//cout<<endl;
		}
	//cout<<"-------------------------------------------------"<<endl;
	}
	offset += batch*m*k;
    }
    //cout<<"OK5.1"<<endl;
    offset = 0;
    for(int p = 0; p < len; p++){
        k = matrix_B_total[p][0].size();
        n = matrix_B_total[p][0][0].size();
        batch = matrix_B_total[p].size();
        for(int i=0; i<batch; i++) {
		for(int t=0; t<n; t++){  
			 for(int j=0; j<k; j++){
				h_B[offset + i*k*n + t*k + j] = matrix_B_total[p][i][j][t];			
			}
		}
	}
	offset += batch*n*k;
    }
    //cout<<"OK5.2"<<endl;
    offset = 0;
    for(int p = 0; p < len; p++){
        m = matrix_C_total[p][0].size();
        n = matrix_C_total[p][0][0].size();
        batch = matrix_C_total[p].size();
	for(int i=0; i<batch; i++) {
		for(int t=0; t<n; t++){
			for(int j=0; j<m; j++){
				c_rel[offset + i*m*n + t*m + j] = matrix_C_total[p][i][j][t];
				h_C[offset + i*m*n + t*m + j] = 0;
			}
		}
	}
	offset += batch*n*m;
    }
    int Zcount = offset;
    //cout<<"OK5.3"<<endl;
    /* =====================================================================
       Performs operation using MAGMABLAS
       =================================================================== */
    magma_setvector(batchCount, sizeof(magma_int_t), h_M, 1, d_M, 1,queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1,queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_K, 1, d_K, 1,queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_lddb, 1, d_lddb, 1, queue );
    magma_setvector(batchCount, sizeof(magma_int_t), h_lddc, 1, d_lddc, 1, queue );
    //cout<<"OK6"<<endl;
    h_A_array[0] = d_A;
    h_B_array[0] = d_B;
    h_C_array[0] = d_C;
    //cout<<"OK6.1"<<endl;
    for (int i = 1; i < batchCount; i++) {
        h_A_array[i] = h_A_array[i-1] + An[i-1] * h_ldda[i-1];
        h_B_array[i] = h_B_array[i-1] + Bn[i-1] * h_lddb[i-1];
        h_C_array[i] = h_C_array[i-1] + h_N[i-1] * h_lddc[i-1];
    }
    //cout<<"OK6.2"<<endl;
    magma_setvector(batchCount, sizeof(float*), h_A_array, 1, d_A_array, 1, queue );
    magma_setvector(batchCount, sizeof(float*), h_B_array, 1, d_B_array, 1, queue );
    magma_setvector(batchCount, sizeof(float*), h_C_array, 1, d_C_array, 1, queue );
    //cout<<"OK7"<<endl;
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
    //cout<<"OK8"<<endl;
    double magma_time;
    magma_time = magma_wtime();
    magmablas_sgemm_vbatched(MagmaNoTrans,MagmaNoTrans,
                     d_M, d_N, d_K,
                     alpha, d_A_array, d_ldda,
                            d_B_array, d_lddb,
                     beta,  d_C_array, d_lddc,
                     batchCount,
                     queue);
    magma_time = magma_wtime() - magma_time;
    cout<<"magma_time: "<<magma_time<<endl;

    //cout<<"OK9"<<endl;
    h_C_tmp = h_Cmagma;
    for (int i = 0; i < batchCount; i++) {
        magma_sgetmatrix( h_M[i], h_N[i], h_C_array[i], h_lddc[i], h_C_tmp, h_ldc[i], queue );
        h_C_tmp += h_N[i] * h_ldc[i];
    }
    //std::cout << "Matrix C after multiplication:" << std::endl;
    //cout<<"OK10"<<endl;
    
    verify(h_Cmagma,c_rel,Zcount);
    
    /*
    for(int k =0 ;k < 2;k++){
    for(int i =0 ;i < 32; i++){
       for(int j=0; j < 32; j++){
           cout<<h_Cmagma[k*1024+i*32 +j]<<" ";
       }
       cout<<endl;
    }
    }
    cout<<"------------------------------------------------"<<endl;
    for(int k =0 ;k < 2;k++){
    for(int i =0 ;i < 32; i++){
       for(int j=0; j < 32; j++){
           cout<<c_rel[k*1024 + i*32 + j]<<" ";
       }
       cout<<endl;
    }
    }
    */
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
    return ;
}
int main(int argc, char *argv[]) {

    vector<vector<vector<vector<float>>>> matrix_A_total,matrix_B_total,matrix_C_total;
    vector<vector<vector<float>>> matrix_A_list,matrix_B_list,matrix_C_list;
    int MMnumber = 0;
    int M_max = 0;
    int K_max = 0;
    int N_max = 0;

    string DS;
    DS = argv[1];
    string path = "/home/changbo/src/daihanwen/Model_Batched_GEMM/DataSets/"+DS+"/result/";
    string prefix_A = "sparseA";
    string prefix_B = "denseB";
    string prefix_C = "denseC";
    vector<string> fileNames_A = getFileNames(path, prefix_A);
    vector<string> fileNames_B = getFileNames(path, prefix_B);
    vector<string> fileNames_C = getFileNames(path, prefix_C);
    sort(fileNames_A.begin(), fileNames_A.end());
    sort(fileNames_B.begin(), fileNames_B.end());
    sort(fileNames_C.begin(), fileNames_C.end());

    //对于列表中的矩阵进行划分，将相同维度的矩阵划分到一起
    //首先构造vector容器存放不同的前缀
    vector<string> pV_A;
    vector<string> pV_B;
    vector<string> pV_C;

    //收集A中不同的前缀
    for(auto filename_A : fileNames_A) {
        string prefix = filename_A.substr(0, 11);
        if (find(pV_A.begin(), pV_A.end(), prefix) == pV_A.end()) {
            //说明是没找到,没找到就放入pV_A中
            pV_A.push_back(prefix);
        }
    }
    for (auto fileprefixA: pV_A) {
        for(auto filename_A : fileNames_A) {
            if (filename_A.substr(0, 11) == fileprefixA) {
                matrix_A_list.push_back(readFile(path + filename_A));
            }
        }
        matrix_A_total.push_back(matrix_A_list);
        matrix_A_list.clear();
    }




    //收集B中不同的前缀
    for(auto filename_B : fileNames_B) {
        string prefix = filename_B.substr(0, 10);
        if (find(pV_B.begin(), pV_B.end(), prefix) == pV_B.end()) {
            //说明是没找到,没找到就放入pV_B中
            pV_B.push_back(prefix);
        }
    }
    for (auto fileprefixB: pV_B) {
        for(auto filename_B : fileNames_B) {
            if (filename_B.substr(0, 10) == fileprefixB) {
                matrix_B_list.push_back(readFile(path + filename_B));
            }
        }
        matrix_B_total.push_back(matrix_B_list);
        matrix_B_list.clear();
    }


    //收集C中不同的前缀
    for(auto filename_C : fileNames_C) {
        string prefix = filename_C.substr(0, 10);
        if (find(pV_C.begin(), pV_C.end(), prefix) == pV_C.end()) {
            //说明是没找到,没找到就放入pV_B中
            pV_C.push_back(prefix);
        }
    }
    for (auto fileprefixC: pV_C) {
        for(auto filename_C : fileNames_C) {
            if (filename_C.substr(0, 10) == fileprefixC) {
                matrix_C_list.push_back(readFile(path + filename_C));
            }
        }
        matrix_C_total.push_back(matrix_C_list);
        matrix_C_list.clear();
    }

    for(int i = 0; i < matrix_A_total.size(); i++){
    	MMnumber += matrix_A_total[i].size();
    	if(i == matrix_A_total.size()-1){
    	    M_max = matrix_A_total[i][0].size();
    	    K_max = M_max;
    	    N_max = matrix_A_total[i][0][0].size();
    	}
    }
    cout<<"MM number: "<<MMnumber<<endl;
    cout<<"M_max: "<<M_max<<endl;
    cout<<"N_max: "<<N_max<<endl;
    
    
    compute(matrix_A_total, matrix_B_total, matrix_C_total,MMnumber, M_max,K_max,N_max);
        

    return 0;
}

