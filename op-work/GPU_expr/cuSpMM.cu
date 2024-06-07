#include <iostream>
//#include <malloc.h>
#include <fstream>
#include <vector>
#include <sstream>
#include "cuda_runtime.h"
//#include "cusparse_v2.h"
//#include "assert.h"
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>        // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <filesystem>
#include <string>
#include <filesystem>
#include <algorithm>
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

//void readFile(ifstream &file,vector<float> &matrix){
//    if (!file.is_open())
//        cout << "Error" << endl;
//    else {
//        string buf1;
//        while (getline(file, buf1)) {
//            stringstream current_line(buf1);
//            float temp1;
//            while (current_line >> temp1) {
//                cout<<temp1<<endl;
//                matrix.push_back(temp1);
//            }
//
//        }
//    }
//    //cout<<matrix.size()<<endl;
//}

//转换为稀疏格式csr
void sparse2csr(float *data, int *&rowPtr, int *&colInd, float*&val, int m, int n){

    rowPtr = (int*)malloc(sizeof(int)*(m + 1));
    int towtal = m * n;
    int* tcolInd = (int*)malloc(sizeof(int)*towtal);
    float* tval = (float*)malloc(sizeof(float)*towtal);

    int nnv = 0;

    for (int i = 0; i < m; i++) {
        rowPtr[i] = nnv;
        for (int j = 0; j < n; j++) {
            int l = i*n + j;
            if (data[l] != 0) {
                tcolInd[nnv] = j;
                tval[nnv] = data[l];
                nnv++;
            }
        }
    }
    rowPtr[m] = nnv;

    colInd = (int*)malloc(sizeof(int)*(nnv));
    val = (float*)malloc(sizeof(float)*(nnv));

    memcpy(colInd, tcolInd, sizeof(int)*nnv);
    memcpy(val, tval, sizeof(float)*nnv);
    free(tcolInd);
    free(tval);
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


float compute(vector<vector<float>>& matrix_A, vector<vector<float>>& matrix_B, vector<vector<float>>& matrix_C){
    int m = matrix_A.size();
    int k = matrix_A[0].size();
    int n = matrix_B[0].size();
    cout<<"M: "<<matrix_A.size()<<endl;
    cout<<"K: "<<matrix_A[0].size()<<endl;
    cout<<"N: "<<matrix_B[0].size()<<endl;
//    for(int i=0;i<m*k;i++){
//        cout<<"matrix_A["<<i<<"]: "<<matrix_A[i]<<endl;
//    }
//    for(int i=0;i<k*n;i++){
//        cout<<"matrix_B["<<i<<"]: "<<matrix_B[i]<<endl;
//    }
//    for(int i=0;i<m*n;i++){
//        cout<<"matrix_C["<<i<<"]: "<<matrix_C[i]<<endl;
//    }

    int ldb=k;
    int ldc=m;

    float *h_A,*h_B,*h_C,*hC_result;
    h_A=(float*)malloc(sizeof(float)*m*k);
    h_B=(float*)malloc(sizeof(float)*n*k);
    h_C=(float*)malloc(sizeof(float)*m*n);
    hC_result=(float*)malloc(sizeof(float)*m*n);

//    下面是测试原始数据集的代码
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

//    cout<<"随机数组准备生成"<<endl;
////  开始生成等维随机矩阵
//    std::random_device rd;  // 随机设备
//    std::mt19937 gen(rd()); // 使用随机设备生成随机种子
//    std::uniform_real_distribution<float> dis(0.0, 1.0);  // 定义均匀分布的范围为（0，1）
//    cout<<"随机种子已经完成"<<endl;
//    for (int i = 0; i < m; i++) {
//        for (int j = 0; j < k; j++) {
//            matrix_A[i][j] = dis(gen);
//            h_A[i*k + j] = matrix_A[i][j];
//        }
//    }
//    cout<<"h_A没有问题"<<endl;
//    for (int i = 0; i < k; i++) {
//        for (int j = 0; j < n; j++) {
//            matrix_B[i][j] = dis(gen);
//            h_B[i*n + j] = matrix_B[i][j];
//        }
//    }
//    cout<<"矩阵C已经清空"<<endl;
//    for (int i = 0; i < m; i++) {
//        for (int j = 0; j < n; j++) {
//            matrix_C[i][j] = 0;
//            for(int t = 0; t < k; t++){
//                matrix_C[i][j] += matrix_A[i][t] * matrix_B[t][j];
//            }
//            hC_result[i*n + j] = matrix_C[i][j];
//        }
//    }
//
//    cout<<"随机数组已经生成完毕"<<endl;


    //初始化h_C数组为全0
    memset(h_C,0, sizeof(float)*m*n);
//    cout<<"h_A:"<<endl;
//    print_matrix_row(h_A,m,k);
//    cout<<"h_B:"<<endl;
//    print_matrix_row(h_B,k,n);
//    print_matrix_row(h_C,m,n);






    float *h_csrVals;  //用来存储稀疏矩阵的非零值个数
    int *h_csrCols;  //用来存储每个值的列索引
    int *h_csrRows;  //用来存储每个值的行索引
    cout<<"开始转化稀疏格式"<<endl;
    sparse2csr(h_A,h_csrRows,h_csrCols,h_csrVals,m,k);
    cout<<"sparse2csr调用成功"<<endl;
    int n_Vals=h_csrRows[m];
/*
    printf("Three array:\n");
    print_matrix_row(h_csrRows, 1, m+1);
    print_matrix_row(h_csrCols, 1, n_Vals);
    print_matrix_row(h_csrVals, 1, n_Vals);
    cout << "M: "<< m << "; N: "<< n <<  "; K: "<< k << endl;
    cout << "factor: " << factor << "; totalNNZ: " << n_Vals <<endl;
*/

    cout<<"开始转置"<<endl;
    transpose(h_B,k,n);


    //printf("Transpose h_B:\n");
    //print_matrix_row(h_B,k,n);
    //printf("h_C:\n");
    //print_matrix_row(h_C,m,n);
//    printf("hC_result before transpose:\n");
//    print_matrix_row(hC_result,m,n);
    transpose(hC_result,m,n);
//    printf("hC_result after transpose:\n");
//    print_matrix_row(hC_result,n,m);
    float alpha = 1.0f;
    float beta = 0.0f;


    float *d_A,*d_B,*d_C;
    float *d_csrVals;
    int *d_csrCols;
    int *d_csrRows;

    cout<<"cuda分配空间"<<endl;
    cudaMalloc((float **)&d_csrVals,sizeof(float) * n_Vals );
    cudaMalloc((int **)&d_csrCols,sizeof(int)* n_Vals );
    cudaMalloc((int **)&d_csrRows,sizeof(int) * (m + 1));
    cudaMalloc((float**)&d_A,sizeof(float)*m*k);
    cudaMalloc((float**)&d_B,sizeof(float)*k*n);
    cudaMalloc((float**)&d_C,sizeof(float)*m*n);
    cudaMemcpy(d_csrVals,h_csrVals,sizeof(float)*n_Vals  ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrCols,h_csrCols,sizeof(int)* n_Vals ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRows,h_csrRows,sizeof(int)* (m + 1) ,cudaMemcpyHostToDevice);



    //cudaMemcpy将生成的矩阵复制到GPU内存中
    cudaMemcpy(d_A,h_A,sizeof(float)*m*k,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,sizeof(float)*k*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C,sizeof(float)*m*n,cudaMemcpyHostToDevice);

    memset(h_B,0, sizeof(float)*k*n);
    cudaMemcpy(h_B, d_B, sizeof(float)* k* n, cudaMemcpyDeviceToHost);
//    printf("h_B:\n");
//    print_matrix_row(h_B,k,n);

    float elapsedTime=0.0f;
    double time=0.0f;
//    double flopsPerSpMM=0.f;
//    double gflops_per_sec=0.f;



    cusparseHandle_t handle{nullptr};
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, m, k, n_Vals,
                      d_csrRows, d_csrCols, d_csrVals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, k, n, ldb, d_B,
                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, m, n, ldc, d_C,
                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    // allocate an external buffer if needed

    cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cout<<"GPU开始计算"<<endl;

    //warm-up
    cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // execute SpMM
    for(int run=0;run<N_RUNS;run++){
        cusparseSpMM(handle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);



    cout<<"GPU结束计算"<<endl;

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    cudaMemcpy(h_C, d_C, sizeof(float)* m* n, cudaMemcpyDeviceToHost);
//    printf("h_C:\n");
//    print_matrix_row(h_C,m,n);


/*
    cudaMemcpy(h_A, d_A, sizeof(float)* m* k, cudaMemcpyDeviceToHost);
    printf("h_A:\n");
    print_matrix_row(h_A,m,k);

    cudaMemcpy(h_B, d_B, sizeof(float)* k* n, cudaMemcpyDeviceToHost);
    printf("h_B:\n");
    print_matrix_row(h_B,k,n);
*/

    //printf("Verfication cusparse result: ");
    //verify(h_C,hC_result,m,n);



    time=elapsedTime/N_RUNS;
//    time/=1.0e3f; //convert time unit from millisecond to second
    float rel = time;

//    flopsPerSpMM=2.0*n_Vals*n;
    //cout<< "flopsPerSpMM: " << flopsPerSpMM<< endl;
//    gflops_per_sec=(flopsPerSpMM/1.0e9f)/time;
//
//    cout << "Kernel Time(ms): " << time*1000.f << endl;
//    cout << "Performance(gflops) = " << gflops_per_sec<< endl;
//
//    cout << endl;

//    printf("%f\n", gflops_per_sec);


    free(h_csrRows);
    free(h_csrCols);
    free(h_csrVals);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_csrVals);
    cudaFree(d_csrCols);
    cudaFree(d_csrRows);
    cudaFree(dBuffer);
    return rel;
}




int main(int argc, char *argv[]) {


    vector<vector<vector<float>>> matrix_A_list,matrix_B_list,matrix_C_list;

    float rel = 0.0;
    string DS;
    cin>>DS;
    string path = "/home/daiwen/model_Batched_GEMM/DataSets/R1/"+DS+"/result/";
    //string path = "/home/daiwen/model_Batched_GEMM/DataSets/"+DS+"/result/";
    string prefix_A = "sparseA";
    string prefix_B = "denseB";
    string prefix_C = "denseC";
    vector<string> fileNames_A = getFileNames(path, prefix_A);
    vector<string> fileNames_B = getFileNames(path, prefix_B);
    vector<string> fileNames_C = getFileNames(path, prefix_C);
    sort(fileNames_A.begin(), fileNames_A.end());
    sort(fileNames_B.begin(), fileNames_B.end());
    sort(fileNames_C.begin(), fileNames_C.end());

    // 循环读取对应的矩阵放入列表
    for(auto filename_A : fileNames_A){
        matrix_A_list.push_back(readFile(path + filename_A));

    }
    
    for(auto filename_B : fileNames_B){
        matrix_B_list.push_back(readFile(path + filename_B));
    }
    
    for(auto filename_C : fileNames_C){
        matrix_C_list.push_back(readFile(path + filename_C));
    }
    int len = fileNames_A.size();
    cout<<"length of list_A:"<<len<<endl;

    for(int i = 0; i < len; i++){
        cout<<"The "<<i<<"th SpMM starts running"<<endl;
        float time = compute(matrix_A_list[i], matrix_B_list[i], matrix_C_list[i]);
        rel += time;
        cout<<"The compute cost is :"<<time<<"ms"<<endl;
        cout<<"The "<<i<<"th SpMM ends running"<<endl;
    }

    cout<<"total cost time is:"<<rel<<"ms";
    return 0;
}



