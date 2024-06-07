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
#define N_RUNS 10
using namespace std;
namespace fs = std::filesystem;

// 函数声明
std::vector<std::vector<float>> readCSV(const std::string& filename);

int main(int argc, char *argv[]) {
    // DS represent DataSet name
    string DS;
    //C++ style:argv[0] = ./program,and argv[1],argv[2]...
    //argc is a int parameter meaning the number of argv(including ./program)
    DS = argv[1];
    // sort file name to get data from .csv in sequence.
    std::string filePathA = "/home/daiwen/model_Batched_GEMM/DataSets/SingleMM/"+DS+"/result/sparseA_"+DS+"_"+DS+"_"+DS+"_1.csv";
    std::string filePathB = "/home/daiwen/model_Batched_GEMM/DataSets/SingleMM/"+DS+"/result/denseB_"+DS+"_"+DS+"_"+DS+"_1.csv";
    std::string filePathC = "/home/daiwen/model_Batched_GEMM/DataSets/SingleMM/"+DS+"/result/denseC_"+DS+"_"+DS+"_"+DS+"_1.csv";

    // 读取CSV文件
    std::vector<std::vector<float>> matrixA = readCSV(filePathA);
    std::vector<std::vector<float>> matrixB = readCSV(filePathB);
    std::vector<std::vector<float>> matrixC = readCSV(filePathC);


    float* A, * B, * C;
    int m, n, k;
    float alpha, beta;


    m = matrixA.size(), k = matrixB.size(), n = matrixC.size();

    alpha = 1.0; beta = 0.0;

    A = (float*)mkl_malloc(m * k * sizeof(float), 64);
    B = (float*)mkl_malloc(k * n * sizeof(float), 64);
    C = (float*)mkl_malloc(m * n * sizeof(float), 64);
    if (A == NULL || B == NULL || C == NULL) {
        cout<<"sth goes wrong."<<endl;
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }


    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++){
        A[i*k+j] = matrixA[i][j];
      }
    }
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++){
        B[i*k+j] = matrixB[i][j];
      }
    }

    for (int i = 0; i < (m * n); i++) {
        C[i] = 0.0;
    }
    //warm-up
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, alpha, A, k, B, n, beta, C, n);

    auto start_time = chrono::high_resolution_clock::now();
    for(int i = 0; i<N_RUNS; i++){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, alpha, A, k, B, n, beta, C, n);
    }
    //计算系统时间，精确到微秒
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time-start_time);
    double duration_us = duration.count();
    double real_time = (double)duration_us/N_RUNS;
    // cout<<"5"<<endl;
    // 输出结果
    cout << "cblas_sgemm cost(us)：" <<real_time<<std::endl;

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}

// 读取CSV文件并返回矩阵
std::vector<std::vector<float>> readCSV(const std::string& filename) {
    std::vector<std::vector<float>> matrix;

    // 打开文件
    std::ifstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return matrix;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::istringstream iss(line);
        std::string value;

        while (std::getline(iss, value, ',')) { // CSV文件中通常是以逗号分隔
            try {
                float num = std::stof(value);
                row.push_back(num);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << std::endl;
            }
        }

        matrix.push_back(row);
    }

    // 关闭文件
    file.close();

    return matrix;
}
