#include <iostream>
//#include "error.cuh"
//#include <malloc.h>
#include <fstream>
#include <vector>
#include <sstream>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <filesystem>
#include <string>
#include <algorithm>
#include <cublas_v2.h>
#include <random>

#define N_RUNS 10
using namespace std;
namespace fs = std::filesystem;

//按行打印
template <typename T>
void print_matrix_row(T *data, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int l = i*n + j;
            cout << data[l] << " ";
        }
        cout << endl;
    }
    cout << endl;
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



//将行读取的矩阵进行转置
void transpose(float *data,int m,int n){
    float *da;
    da= (float*)malloc(sizeof(float)*m*n);
    int count=0;
//    for(int i=0;i<n;i++){
//        for(int j=0;j<m;j++){
//            da[count]=data[j*n+i];
//            count++;
//        }
//    }
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            da[count]=data[j*n+i];
            count++;
        }
    }
    memcpy(data,da,sizeof(float)*m*n);
    free(da);
}

//验证实验结果
void verify(float* matrix1,float* matrix2,int m,int n){
    for(int i=0;i<m*n;i++){
        if(abs(matrix1[i]-matrix2[i])>1.0e-2){
            cout << "The result of matrix multiplication is wrong!" << endl;
            return;
        }

    }
    cout <<  "The result of matrix multiplication is true!" << endl;
}




float compute(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B, const vector<vector<float>>& matrix_C)
{
    int m = matrix_A.size();
    int k = matrix_A[0].size();
    int n = matrix_B[0].size();
    cout<<"M: "<<matrix_A.size()<<endl;
    cout<<"K: "<<matrix_A[0].size()<<endl;
    cout<<"N: "<<matrix_B[0].size()<<endl;

//    int ldb=k;
//    int ldc=m;

    float *h_A,*h_B,*h_C,*hC_result;
    h_A=(float*)malloc(sizeof(float)*m*k);
    h_B=(float*)malloc(sizeof(float)*k*n);
    h_C=(float*)malloc(sizeof(float)*m*n);
    hC_result=(float*)malloc(sizeof(float)*m*n);

//    原数组数据计算
    cout<<"开始存储矩阵到一维数组"<<endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i*k + j] = matrix_A[i][j];
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_B[i*n + j] = matrix_B[i][j];
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            hC_result[i*n + j] = matrix_C[i][j];
        }
    }



    //初始化h_C数组为全0
    memset(h_C,0, sizeof(float)*m*n);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
 //transpose
    transpose(h_B,k,n);
    transpose(hC_result,m,n);

    // Allocate device memory
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    float elapsedTime=0.0f;
    double time=0.0f;

    //warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // execute SpMM
    for(int run=0;run<N_RUNS;run++){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    // Perform matrix multiplication
    
    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Verfication cusparse result: ");
    /*
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout<<hC_result[i*n + j]<<endl;
            
        }
    }
    cout<<"-----------------------------------------"<<endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            
            cout<<h_C[i*n+j]<<endl;
        }
    }*/       
    verify(h_C,hC_result,m,n);
    //printf("ERROR text: %s\n",cudaGetErrorString(error_code));


    time=elapsedTime/N_RUNS;

    float rel = time;


    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return rel;
}

int main(int argc, char *argv[]) {


    vector<vector<vector<float>>> matrix_A_list,matrix_B_list,matrix_C_list;

    float rel = 0.0;
    string DS;
    cin>>DS;
    string path = "/home/daiwen/model_Batched_GEMM/DataSets/"+DS+"/result/";
    string prefix_A = "sparseA";
    string prefix_B = "denseB";
    string prefix_C = "denseC";
    vector<string> fileNames_A = getFileNames(path, prefix_A);
    vector<string> fileNames_B = getFileNames(path, prefix_B);
    vector<string> fileNames_C = getFileNames(path, prefix_C);

    for(auto filename_A : fileNames_A){
    matrix_A_list.push_back(readFile(path + filename_A));
    }
    for(auto filename_B : fileNames_B){
    matrix_B_list.push_back(readFile(path + filename_B));
    }
    for(auto filename_C : fileNames_C){
    matrix_C_list.push_back(readFile(path + filename_C));
    }

    float time = compute(matrix_A_list[0], matrix_B_list[0], matrix_C_list[0]);
        rel += time;
        cout<<"The compute cost is :"<<time<<endl;
        
    

    return 0;
}
