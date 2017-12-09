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

int no_of_pts = 50;
double a = 1;
double dx = a/double(no_of_pts);
int N = 100; //no of unit cells
double init_pt= 0.0;
double final_pt = 10.0;
double A=1;

ofstream dataout("data.txt");
double V(double x){return -A*(1+cos(2*M_PI*x/a));}
double Sqr(cd x){return (x*conj(x)).real();}

bool diagonalize(MatrixXcd A, VectorXcd& lambda, MatrixXcd& v)
{
  int N = A.cols();
  if (A.rows()!=N)  return false;
  v.resize(N,N);
  lambda.resize(N);

  int LDA = A.outerStride();
  int LDV = v.outerStride();
  int INFO = 0;
  cd* w = const_cast<cd*>(lambda.data());
  char Nchar = 'N';
  char Vchar = 'V';
  int LWORK = int(A.size())*4;
  VectorXcd WORK(LWORK);
  VectorXd RWORK(2*LDA);

  zgeev_(&Nchar, &Vchar, &N, reinterpret_cast <__complex__ double*> (A.data()), &LDA, reinterpret_cast <__complex__ double*> (w), 0, &LDV, reinterpret_cast <__complex__ double*> (v.data()), &LDV,  reinterpret_cast <__complex__ double*> (WORK.data()), &LWORK, RWORK.data(), &INFO);

  for(int i=0; i<N; i++)
 	 v.col(i)=v.col(i)/v.col(i).unaryExpr(&Sqr).sum();

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
  MatrixXcd H(no_of_pts,no_of_pts);
  VectorXd point(no_of_pts);

  for(int i=0; i<point.size(); i++) point(i) = i*dx;

  vector < pair<double,VectorXcd> > eigenspectrum;
  VectorXcd v; MatrixXcd eigenvectors; VectorXd eigenvalues;

  for(int n=0; n<N; n++)
  {
    double k = (2*M_PI*n)/(N*a)-M_PI/a;
    for(int i=0; i<point.size(); i++)
    {
      int j = (i==point.size()-1)? 0 : i+1;
      H(i,j)= cd(-1/(2*dx*dx), -k/(2*dx));
      H(j,i)= cd(-1/(2*dx*dx), k/(2*dx));
      H(i,i)= cd(1/(dx*dx)+ pow(k,2)/2+ V(point(i)), 0);
    }

    // cout << H << endl;
    // cout << endl << point.transpose() << endl;


    // ComplexEigenSolver <MatrixXcd> ces; ces.compute(H);
    // eigenvalues = ces.eigenvalues().real(); eigenvalues.resize(5);

    diagonalize(H,v,eigenvectors);
    eigenvalues = sortascending(v.real());

    dataout << k << " " << eigenvalues.transpose() << endl;
  }

  return 0;
}


// eigenspectrum.clear();
// for(int i=0; i<point.size(); i++)  eigenspectrum.push_back(make_pair(ces.eigenvalues()[i].real(), ces.eigenvectors().col(i)));
// sort(eigenspectrum.begin(),eigenspectrum.end(),compare);
// eigenspectrum.resize(5);
