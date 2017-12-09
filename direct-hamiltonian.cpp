#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <complex>
#include <lapacke.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace std::literals;

typedef complex<double> cd;

ofstream dataout("data_direct.txt");

int no_of_pts = 1000;
double a = 1.0;
int N = 100; //no of unit cells
double init_pt= 0.0;
double final_pt = N*a;
double dx = N*a/double(no_of_pts);

double A=1;
double V(double x){return -A*(1+cos(2*M_PI*x/a));}
double Sqr(cd x){return (x*conj(x)).real();}



bool diagonalize(MatrixXcd Ac, VectorXcd& lambdac, MatrixXcd& vc)
{
  int N;
  if(Ac.cols()==Ac.rows())  N = Ac.cols(); else return false;

  MatrixXd A = Ac.real();
  lambdac.resize(N);
  vc.resize(N,N);
  VectorXd lambda = lambdac.real();

  int LDA = A.outerStride();
  int INFO = 0;
  char Uchar = 'U';
  char Vchar = 'V';
  int LWORK = 5*(2*LDA*LDA+6*LDA+1);
  int LIWORK = 5*(3+5*LDA);

  VectorXd WORK(LWORK);
  VectorXi IWORK(IWORK);

  dsyevd_(&Vchar, &Uchar, &N, A.data(), &LDA, lambda.data(),  WORK.data(), &LWORK, IWORK.data(), &LIWORK, &INFO);
  vc.real() = A;
  lambdac.real() = lambda;
  return INFO==0;
}

VectorXd sortascending(VectorXd v1)
{
  vector<double> stdv1 (v1.data(),v1.data()+v1.size());
  std::sort (stdv1.begin(), stdv1.end());
  Map<ArrayXd> sorted(stdv1.data(), stdv1.size());
  return sorted;
}


int main()
{
  VectorXd point(no_of_pts);

  for(int i=0; i<point.size(); i++) point(i) = i*dx;

  vector < pair<double,VectorXcd> > eigenspectrum;
  VectorXcd v; MatrixXcd eigenvectors; VectorXd eigenvalues;

  MatrixXcd H = MatrixXcd::Zero(point.size(),point.size());
  for(int i=0; i<point.size(); i++)
  {
      int j = (i==point.size()-1)? 0 : i+1;
      H(i,j)= -1/(2*dx*dx);
      H(j,i)= -1/(2*dx*dx);
      H(i,i) = 1/(dx*dx)+ V(point(i));
  }

    // ComplexEigenSolver <MatrixXcd> ces; ces.compute(H);
    // eigenvalues = ces.eigenvalues().real(); eigenvalues.resize(5);

    diagonalize(H,v,eigenvectors);
    eigenvalues = sortascending(v.real());

    dataout << eigenvalues << endl;


  return 0;
}


// bool diagonalize(MatrixXcd A, VectorXcd& lambda, MatrixXcd& v)
// {
//   int N = A.cols();
//   if (A.rows()!=N)  return false;
//   v.resize(N,N);
//   lambda.resize(N);
//
//   int LDA = A.outerStride();
//   int LDV = v.outerStride();
//   int INFO = 0;
//   cd* w = const_cast<cd*>(lambda.data());
//   char Nchar = 'N';
//   char Vchar = 'V';
//   int LWORK = int(A.size())*4;
//   VectorXcd WORK(LWORK);
//   VectorXd RWORK(2*LDA);
//
//   zgeev_(&Nchar, &Vchar, &N, reinterpret_cast <__complex__ double*> (A.data()), &LDA, reinterpret_cast <__complex__ double*> (w), 0, &LDV, reinterpret_cast <__complex__ double*> (v.data()), &LDV,  reinterpret_cast <__complex__ double*> (WORK.data()), &LWORK, RWORK.data(), &INFO);
//
//   for(int i=0; i<N; i++)
//  	 v.col(i)=v.col(i)/v.col(i).unaryExpr(&Sqr).sum();
//
//   return INFO==0;
// }
