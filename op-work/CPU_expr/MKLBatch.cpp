#include <iostream>
#include <stdio.h>
#include <mkl.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdlib.h>           // EXIT_FAILURE
#include <filesystem>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#define N_RUNS 100
using namespace std;
namespace fs = std::filesystem;

/*
2023.11.21 Daiwen
Intel MKL BatchGEMM realization
-----------------------------------
Range of application:
/home/daiwen/model_Batched_GEMM/DataSets/"+DS+"/result/

Utilization:
open terminal in corresponding path of this file
. /opt/intel/oneapi/setvars.sh
g++ MKLBatch.cpp -o MKLBatch.out -lmkl_rt -std=c++17
./MKLBatch.out
D1/D2/...
-----------------------------------
MKL Kernel Function:
cblas_sgemm_batch();

-----------------------------------
Self-defining Function:

getFileNames():
from main(),
Get csv file name and store into fileNames(vector).
This is for further operation.

readFiles()：
from main(),
read from .csv file to data(vector<vector<type>>)
following the filename.

verify():
from compute(),
verify the correctness of cblas_sgemm_batch().

compute():
kernel compute function.

main();
prepare the data.

See details in main() and compute().
*/


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

void verify(float* c_array[],float* c_rel[],MKL_INT Size){
  //c_array is the compute result.
  //c_rel is the real answer from matrixC_total
    for(int i=0;i<Size;i++){
        if(abs(*c_array[i]-*c_rel[i])>1.0e-2){
          cout << "The result of matrix multiplication is wrong!" << endl;
            return;
        }
    }
    cout <<  "The result of matrix multiplication is true!" << endl;
}

void compute(MKL_INT GRP_COUNT, MKL_INT size_per_grp[],MKL_INT m[],
   MKL_INT k[], MKL_INT n[],MKL_INT lda[],MKL_INT ldb[],MKL_INT ldc[],
 vector<vector<vector<vector<float>>>>& matrix_A_total,
 vector<vector<vector<vector<float>>>>& matrix_B_total,
 vector<vector<vector<vector<float>>>>& matrix_C_total,MKL_INT Size) {

    //transpose or not
    CBLAS_TRANSPOSE    transA[GRP_COUNT];
    CBLAS_TRANSPOSE    transB[GRP_COUNT];

    //C = alpha*A*B + beta*C
    float alpha[GRP_COUNT];
    float beta[GRP_COUNT];
    for(int i = 0;i < GRP_COUNT;i++){
      transA[i] = CblasNoTrans;
      transB[i] = CblasNoTrans;
      alpha[i] = 1.0;
      beta[i] = 0.0;
    }

    // one-D array for data storage
    float  *a_array[Size], *b_array[Size], *c_array[Size], *c_rel[Size];
    int t = 0;
    int size_a = 0;
  	int size_b = 0;
  	int size_c = 0;
    for(int q = 0; q < GRP_COUNT; q++){


      cout<<"The "<<q<<"th Group info:"<<endl;
      cout<<"m["<<q<<"]:"<<m[q]<<endl;
      cout<<"k["<<q<<"]:"<<k[q]<<endl;
      cout<<"n["<<q<<"]:"<<n[q]<<endl;

    //size_a means the number of one A matrix elements
      size_a = m[q]*k[q];
      //cout<<"size_a:"<<size_a<<endl;
      size_b = k[q]*n[q];
      //cout<<"size_b:"<<size_b<<endl;
      size_c = m[q]*n[q];
      //cout<<"size_c:"<<size_c<<endl;
      int sg = size_per_grp[q];
      cout<<"size_per_grp["<<q<<"]:"<<size_per_grp[q]<<endl;

      float* a[sg];
    	float* b[sg];
    	float* c1[sg];
      float* c2[sg];

    //Key!:the storage process
      for(int r = 0; r < sg; r++){
        a[r] = new float[size_a];
        for(int i = 0; i < m[q]; i++){
          for(int j = 0; j < k[q]; j++){
            a[r][i*k[q] + j] = matrix_A_total[q][r][i][j];

          }
        }
      }
      //cout<<"OK2.2"<<endl;
      for(int r = 0; r < size_per_grp[q]; r++){
        b[r] = new float[size_b];
        for(int i = 0; i < k[q]; i++){
          for(int j = 0; j < n[q]; j++){
            b[r][i*n[q] + j] = matrix_B_total[q][r][i][j];
          }
        }
      }
      // cout<<"OK2.3"<<endl;
      for(int r = 0; r < size_per_grp[q]; r++){
        c1[r] = new float[size_c];
        c2[r] = new float[size_c];
        for(int i = 0; i < m[q]; i++){
          for(int j = 0; j < n[q]; j++){
            c1[r][i*n[q] + j] = 0;
            c2[r][i*n[q] + j] = matrix_C_total[q][r][i][j];
          }
        }
      }
      // cout<<"OK2.4"<<endl;
      int tmp = 0;
      for(int i = t; i < t+size_per_grp[q]; i++){
        a_array[i] = a[tmp];
        b_array[i] = b[tmp];
        c_array[i] = c1[tmp];
        c_rel[i] = c2[tmp];
        tmp++;
      }
      // cout<<"OK2.5"<<endl;
      t += size_per_grp[q];
    }
    // cout<<"OK3"<<endl;
    // Call cblas_sgemm_batch
    //warm-up
    cblas_sgemm_batch (
            CblasRowMajor,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            (const float**)a_array,
            lda,
            (const float**)b_array,
            ldb,
            beta,
            c_array,
            ldc,
            GRP_COUNT,
            size_per_grp);
    // cout<<"OK4"<<endl;
    //recoed system clock
    vector<double> duration_us(N_RUNS,0);
    for(int i = 0; i<N_RUNS; i++){
      auto start_time = chrono::high_resolution_clock::now();
      cblas_sgemm_batch (
            CblasRowMajor,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            (const float**)a_array,
            lda,
            (const float**)b_array,
            ldb,
            beta,
            c_array,
            ldc,
            GRP_COUNT,
            size_per_grp);
      auto end_time = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::nanoseconds>(end_time-start_time);
      duration_us[i] = duration.count();
    }
    //计算系统时间，精确到微秒
    double max_value = *std::max_element(duration_us.begin(),duration_us.end());
    double min_value = *std::min_element(duration_us.begin(),duration_us.end());
    sort(duration_us.begin(),duration_us.end());
    double median_value = (N_RUNS%2 == 0) ? (duration_us[N_RUNS/2-1] + duration_us[N_RUNS/2])/2 : duration_us[N_RUNS/2];
    size_t index75_percent = static_cast<size_t>(N_RUNS * 0.75);
    size_t index25_percent = static_cast<size_t>(N_RUNS * 0.25);
    double value75_percent = duration_us[index75_percent];
    double value25_percent = duration_us[index25_percent];

    //输出结果
    cout<<"-------------------------------"<<endl;
    for(int i = 0; i < N_RUNS; i++){
      cout<<i<<" : "<<(duration_us[i]/1000)<<endl;
    }
    cout<<"--------------------------------"<<endl;
    cout<<"cost(us):"<<endl;
    cout<<"max:"<<(max_value/1000)<<endl;
    cout<<"index75:"<<(value75_percent/1000)<<endl;
    cout<<"median:"<<(median_value/1000)<<endl;
    cout<<"index25:"<<(value25_percent/1000)<<endl;
    cout<<"min:"<<(min_value/1000)<<endl;

    // cout<<"5"<<endl;
    // 输出结果
    //cout << "cblas_sgemm_batch cost(us)：" <<real_time<<std::endl;
    verify(c_array,c_rel,Size);
    return ;
}
int main(int argc, char *argv[]){
    /*
       Use vector for dynamic storage.
       matrix_A_total means total data divided by dimension
       matrix_A_list stores all the matrix in same dimension
       Actually
       matrix_A_total = {matrix_A_list1, matrix_A_list2,......}
    */
    vector<vector<vector<vector<float>>>> matrix_A_total,matrix_B_total,matrix_C_total;
    vector<vector<vector<float>>> matrix_A_list,matrix_B_list,matrix_C_list;

    // DS represent DataSet name
    string DS;
    //C++ style:argv[0] = ./program,and argv[1],argv[2]...
    //argc is a int parameter meaning the number of argv(including ./program)
    DS = argv[1];
    // sort file name to get data from .csv in sequence.
    string path = "/home/daiwen/model_Batched_GEMM/DataSets/"+DS+"/result/";
    string prefix_A = "sparseA";
    string prefix_B = "denseB";
    string prefix_C = "denseC";
    vector<string> fileNames_A = getFileNames(path, prefix_A);
    vector<string> fileNames_B = getFileNames(path, prefix_B);
    vector<string> fileNames_C = getFileNames(path, prefix_C);
    MKL_INT Size = fileNames_A.size();
    cout<<"Count A matrix:"<<Size<<endl;
    sort(fileNames_A.begin(), fileNames_A.end());
    sort(fileNames_B.begin(), fileNames_B.end());
    sort(fileNames_C.begin(), fileNames_C.end());

    //realize matrix_A_total = {matrix_A_list1, matrix_A_list2,......}
    //store different prefix in vectors.
    vector<string> pV_A;
    vector<string> pV_B;
    vector<string> pV_C;

    //get different prefix in A
    for(auto filename_A : fileNames_A) {
        string prefix = filename_A.substr(0, 11);
        if (find(pV_A.begin(), pV_A.end(), prefix) == pV_A.end()) {
            //put new prefix into pV_A
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

    //get different prefix in B
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


    //get different prefix in C
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


    //GRP_COUNT means number of Kind of M dimension in one DS.
    MKL_INT    GRP_COUNT = matrix_A_total.size();
    cout<<"The Number of dimension variety is :"<<GRP_COUNT<<endl;

    //these arrays record the dimension of different group.
    //size_per_grp record the number of matrix in different group.
    MKL_INT    m[GRP_COUNT];
    MKL_INT    k[GRP_COUNT];
    MKL_INT    n[GRP_COUNT];
    MKL_INT    size_per_grp[GRP_COUNT];

    for(int i = 0; i < GRP_COUNT; i++){
        size_per_grp[i] = matrix_A_total[i].size();
        m[i] = matrix_A_total[i][0].size();
        k[i] = matrix_A_total[i][0][0].size();
        n[i] = matrix_B_total[i][0][0].size();
    }

    //lda,ldb,ldc represent the reading way
    //number of reading data in one time
    MKL_INT    lda[GRP_COUNT];
    MKL_INT    ldb[GRP_COUNT];
    MKL_INT    ldc[GRP_COUNT];
    for(int i = 0; i < GRP_COUNT; i++){
        lda[i] = k[i];
        ldb[i] = n[i];
        ldc[i] = n[i];
    }

    compute(GRP_COUNT,size_per_grp,m,k,n,lda,ldb,ldc,
    matrix_A_total,matrix_B_total,matrix_C_total,Size);

    return 0;

}
