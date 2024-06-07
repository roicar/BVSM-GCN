#include <iostream>
#include <mkl.h>
#define    GRP_COUNT    2
using namespace std;
int main() {
    MKL_INT    m[GRP_COUNT] = {2, 3};
    MKL_INT    k[GRP_COUNT] = {2, 2};
    MKL_INT    n[GRP_COUNT] = {3, 2};

    MKL_INT    lda[GRP_COUNT] = {2, 2};
    MKL_INT    ldb[GRP_COUNT] = {3, 2};
    MKL_INT    ldc[GRP_COUNT] = {3, 2};

    CBLAS_TRANSPOSE    transA[GRP_COUNT] = {CblasNoTrans, CblasNoTrans};
    CBLAS_TRANSPOSE    transB[GRP_COUNT] = {CblasNoTrans, CblasNoTrans};

    double    alpha[GRP_COUNT] = {1.0, 1.0};
    double    beta[GRP_COUNT] = {0.0, 0.0};

    MKL_INT    size_per_grp[GRP_COUNT] = {2, 2};
    cout<<"1"<<endl;
    // Total number of multiplications: 4
    double    *a_array[4], *b_array[4], *c_array[4];
    cout<<"1.5"<<endl;
    double a1[] = {1.0,2.0,3.0,4.0};
    double a2[] = {1.0,2.0,3.0,4.0};
    double a3[] = {1.0,2.0,3.0,4.0,5.0,6.0};
    double a4[] = {1.0,2.0,3.0,4.0,5.0,6.0};
    a_array[0] = a1;
    a_array[1] = a2;
    a_array[2] = a3;
    a_array[3] = a4;
    cout<<"2"<<endl;
    double b1[] = {1.0,2.0,3.0,4.0,5.0,6.0};
    double b2[] = {1.0,2.0,3.0,4.0,5.0,6.0};
    double b3[] = {1.0,2.0,3.0,4.0};
    double b4[]= {1.0,2.0,3.0,4.0};
    b_array[0] = b1;
    b_array[1] = b2;
    b_array[2] = b3;
    b_array[3] = b4;
    cout<<"3"<<endl;
    double c1[] = {0.0,0.0,0.0,0.0,0.0,0.0};
    double c2[] = {0.0,0.0,0.0,0.0,0.0,0.0};
    double c3[] = {0.0,0.0,0.0,0.0,0.0,0.0};
    double c4[] = {0.0,0.0,0.0,0.0,0.0,0.0};
    c_array[0] = c1;
    c_array[1] = c2;
    c_array[2] = c3;
    c_array[3] = c4;
    cout<<"4"<<endl;
    // Call cblas_dgemm_batch
    cblas_dgemm_batch (
            CblasRowMajor,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            (const double**)a_array,
            lda,
            (const double**)b_array,
            ldb,
            beta,
            c_array,
            ldc,
            GRP_COUNT,
            size_per_grp);

    cout<<"5"<<endl;
    // 输出结果
    std::cout << "矩阵乘法结果：" << std::endl;
    int b=0;
      for(int i = 0; i < m[b]; i++){
        for(int j = 0; j < n[b]; j++){
            cout << c_array[0][i * n[b] + j] << " ";
        }
        cout<<endl;
      }
      for(int i = 0; i < m[b]; i++){
        for(int j = 0; j < n[b]; j++){
            cout << c_array[1][i * n[b] + j] << " ";
        }
        cout<<endl;
      }
    b=1;
    for(int i = 0; i < m[b]; i++){
        for(int j = 0; j < n[b]; j++){
            cout << c_array[2][i * n[b] + j] << " ";
        }
        cout<<endl;
      }
    for(int i = 0; i < m[b]; i++){
      for(int j = 0; j < n[b]; j++){
          cout << c_array[3][i * n[b] + j] << " ";
      }
      cout<<endl;
    }



    return 0;
}
