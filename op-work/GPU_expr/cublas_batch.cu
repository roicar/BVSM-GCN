#include <iostream>
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
#include <assert.h>

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

__global__
void show(float* ptr, int size)
{
        for(int i =0; i<size; i++)
        printf("%f\n", ptr[i]);
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
void verify(float* matrix1,float* matrix2,int m,int n, int batch){
    for(int i=0;i<m*n;i++){
        if(abs(matrix1[i]-matrix2[i])>1.0e-2){
            cout << "The result of matrix multiplication is wrong!" << endl;
            return;
        }

    }
    cout <<  "The result of matrix multiplication is true!" << endl;
}

float compute(vector<vector<vector<float>>>& matrix_A_list, vector<vector<vector<float>>>& matrix_B_list, vector<vector<vector<float>>>& matrix_C_list)
{
	//cout<<"0"<<endl;
	int m = matrix_A_list[0].size();
	int k = matrix_A_list[0][0].size();
	int n = matrix_B_list[0][0].size();
	int batch = matrix_A_list.size();
	cout<<"M: "<<m<<endl;
	cout<<"K: "<<k<<endl;
	cout<<"N: "<<n<<endl;
	cout<<"batch_size: "<<batch<<endl;

	int size_a = m*k*batch;
	int size_b = k*n*batch;
	int size_c = m*n*batch;
	float* a = new float[size_a];
	float* b = new float[size_b];
	float* c_rel = new float[size_c];
	float* c = new float[size_c];
	
	
	cout<<"随机数组准备生成"<<endl;
//  开始生成等维随机矩阵
	std::random_device rd;  // 随机设备
	std::mt19937 gen(rd()); // 使用随机设备生成随机种子
	std::uniform_real_distribution<float> dis(0.0, 1.0);  // 定义均匀分布的范围为（0，1）
	cout<<"随机种子已经完成"<<endl;
	for (int i = 0; i < batch; i++) {
	for (int j = 0; j < m; i++) {
	for (int t = 0; t < k; j++) {
	    matrix_A_list[i][j][t] = dis(gen);
	}
	}
	}
	cout<<"h_A没有问题"<<endl;
	for (int i = 0; i < batch; i++) {
	for (int j = 0; j < k; i++) {
	for (int t = 0; t < n; j++) {
	    matrix_B_list[i][j][t] = dis(gen);
	}
	}
	}
	cout<<"矩阵C已经清空"<<endl;
	for (int p = 0; p < batch; p++) {
	for (int i = 0; i < m; i++) {
	for (int j = 0; j < n; j++) {
	    matrix_C_list[p][i][j] = 0;
	    for(int t = 0; t < k; t++){
		matrix_C_list[p][i][j] += matrix_A_list[p][i][t] * matrix_B_list[p][t][j];
	    }
	}
	}
	}
	cout<<"随机数组已经生成完毕"<<endl;
	
	for(int i=0; i<m; i++) {
		for(int j=0; j<batch; j++){
			for(int t=0; t<k; t++){
				a[i*batch*k + j*k + t] = matrix_A_list[j][i][t];
			
			}
		}
	}
	//cout<<"1"<<endl;
	for(int i=0; i<k; i++) {
		for(int j=0; j<batch; j++){
			for(int t=0; t<n; t++){
				b[i*batch*n + j*n + t] = matrix_B_list[j][i][t];
			
			}
		}
	}
	//for(int i = 0; i< batch*k*n;i++)
	//		cout<<b[i]<<endl;
	//cout<<"2"<<endl;
	for(int i=0; i<m; i++) {
		for(int j=0; j<batch; j++){
			for(int t=0; t<n; t++){
				c_rel[i*batch*n + j*n + t] = matrix_C_list[j][i][t];
				c[i*batch*n + j*n + t] = 3.0;
			}
		}
	}
	//for(int i = 0; i< batch*m*n;i++)
	//		cout<<c_rel[i]<<endl;
	//cout<<"3"<<endl;
	




	float* d_a,* d_b,*d_c;

	cudaMalloc((void**)&d_a, batch * m * k * sizeof(float));
	cudaMalloc((void**)&d_b, batch * k * n * sizeof(float));
	cudaMalloc((void**)&d_c, batch * m * n * sizeof(float));

	cudaMemcpy(d_a, a, batch * m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, batch * k * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, batch * m * n * sizeof(float), cudaMemcpyHostToDevice);
	
	
	//cout<<"2"<<endl;
	cublasHandle_t handle;
        cublasStatus_t ret;
        ret = cublasCreate(&handle);
        float *a_array[batch], *b_array[batch];
        float *c_array[batch];
        for (int j = 0; j < batch; ++j) {
                a_array[j] = d_a + j * k;
                b_array[j] = d_b + j * n;
                c_array[j] = d_c + j * n;
        }
        /*
        for (int j = 0; j < batch; ++j) {
        	cout<<"b_array:"<<endl;
        	show<<<1,1>>>(b_array[j], 1);
        	cout<<"a_array:"<<endl;
        	show<<<1,1>>>(a_array[j], 1);
        	cout<<"c_array:"<<endl;
        	show<<<1,1>>>(c_array[j], 1);

        }*/
  	
        const float **d_Marray, **d_Narray;
        float **d_Parray;
        cudaMalloc((void**)&d_Marray, batch*sizeof(float *));
        cudaMalloc((void**)&d_Narray, batch*sizeof(float *));
        cudaMalloc((void**)&d_Parray, batch*sizeof(float *));
        cudaMemcpy(d_Marray, a_array, batch*sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Narray, b_array, batch*sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Parray, c_array, batch*sizeof(float *), cudaMemcpyHostToDevice);
	
	
	//cout<<"3"<<endl;
	
	
	const float alpha = 1.0f;
	const float beta = 0.0f;

	float elapsedTime=0.0f;
	double time=0.0f;
	
	int lda=batch*k;
	int ldb=batch*n;
	int ldc=batch*n;

	// 执行批处理矩阵乘法 warm-up
	cublasSgemmBatched(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           n,m,k,
                           &alpha,
                           d_Narray,  ldb,
                           d_Marray,  lda,
                           &beta,
                           d_Parray,  ldc,
                           batch);
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	//cout<<"4"<<endl;
	for(int i = 0; i<N_RUNS; i++){
	ret = cublasSgemmBatched(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           n,m,k,
                           &alpha,
                           d_Narray,  ldb,
                           d_Marray,  lda,
                           &beta,
                           d_Parray,  ldc,
                           batch);
                           
        }
	cudaEventRecord(stop,0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cublasDestroy(handle);
        if (ret == CUBLAS_STATUS_SUCCESS)
        {
        printf("sgemm success  %d, line(%d)\n", ret, __LINE__);
        }
        cudaMemcpy(c, d_c, batch * m * n * sizeof(float), cudaMemcpyDeviceToHost);
	
	//cout<<"5"<<endl
	printf("Verfication batch_cublas result: ");	
	verify(c,c_rel,m,n,batch);
	

	time=elapsedTime/N_RUNS;
	float rel = time;
	
	

        free(a);
        free(b);
        free(c);
    	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_Marray);
	cudaFree(d_Narray);
	cudaFree(d_Parray);
	
	return rel;
}
int main(int argc, char *argv[]) {

    vector<vector<vector<vector<float>>>> matrix_A_total,matrix_B_total,matrix_C_total;
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


    //len表示不同维度的矩阵列表的个数
    int len = matrix_A_total.size();
    cout<<"len is :"<<len<<endl;

    for(int i = 0; i < len; i++){
        cout<<"The "<<i<<"th cublas starts running"<<endl;
        float time = compute(matrix_A_total[i], matrix_B_total[i], matrix_C_total[i]);
        rel += time;
        cout<<"The compute cost is :"<<time<<endl;
        cout<<"The "<<i<<"th cublas ends running"<<endl;
    }

    cout<<"total cost time is:"<<rel;
    return 0;
}
