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
typedef vector < pair<double,VectorXcd> > spectrum;

int no_of_unitcell_pts=50;
double a = 1;
double dx = a/double(no_of_unitcell_pts);
int N = 20; //no of unit cells
int no_of_total_pts = no_of_unitcell_pts*N;
double init_pt= 0.0;
double final_pt = 10.0;
double A=1;
double epsilon = dx/2;

spectrum* arr = new spectrum [N];
VectorXd lattice_point(no_of_unitcell_pts);
ofstream dataout("data_perturb.txt");

double V(double x){return -A*(1+cos(2*M_PI*x/a));}
double Sqr(cd x){return (x*conj(x)).real();}
bool compare(const pair<double, VectorXcd>&i, const pair<double, VectorXcd>&j) {return i.first < j.first;}
cd u(int m, int k, int i) { return arr[k].at(m).second(i%N);}

double direct_core_integrand(int m, int k, int kappa, int i, int i_prime)
{
  return norm(u(m,kappa,i_prime))/(abs(lattice_point(i)-lattice_point(i_prime))+epsilon);
}

double direct_integral_prime(int m, int k, int kappa, int i)
{
  double trapez_sum=0.0;
  double fa= direct_core_integrand(m,k,kappa,i,0)/2.0;
  double fb= direct_core_integrand(m,k,kappa,i,lattice_point.size()-1)/2.0;
  for(int i_prime=1; i_prime < lattice_point.size()-1; i_prime++) trapez_sum+= direct_core_integrand(m,k,kappa,i,i_prime);
  return norm(u(m,k,i))*(trapez_sum+fb+fa);
}

double direct_integral(int m, int k, int kappa)
{
  double fa= direct_integral_prime(m,k,kappa,0)/2.0;
  double fb= direct_integral_prime(m,k,kappa,lattice_point.size()-1)/2.0;
  double trapez_sum=0.0;
  for(int i=1; i< lattice_point.size()-1; i++) trapez_sum+= direct_integral_prime(m,k,kappa,i);
  return (trapez_sum+fb+fa);
}

cd exchange_core_integrand(int m, int k, int kappa, int i, int i_prime)
{
  return conj(u(m,k,i_prime))*u(m,kappa,i_prime)*exp(cd(0,(k-kappa)*(lattice_point(i)-lattice_point(i_prime))))/(abs(lattice_point(i)-lattice_point(i_prime))+epsilon);
}

cd exchange_integral_prime(int m, int k, int kappa, int i)
{
  cd trapez_sum=cd(0,0);
  cd fa= exchange_core_integrand(m,k,kappa,i,0)/2.0;
  cd fb= exchange_core_integrand(m,k,kappa,i,lattice_point.size()-1)/2.0;
  for(int i_prime=1; i_prime < lattice_point.size()-1; i_prime++) trapez_sum+= exchange_core_integrand(m,k,kappa,i,i_prime);
  return conj(u(m,kappa,i))*u(m,k,i)*(trapez_sum+fb+fa);
}

cd exchange_integral(int m, int k, int kappa)
{
  cd fa= exchange_integral_prime(m,k,kappa,0)/2.0;
  cd fb= exchange_integral_prime(m,k,kappa,lattice_point.size()-1)/2.0;
  cd trapez_sum=cd(0,0);
  for(int i=1; i< lattice_point.size()-1; i++) trapez_sum+= exchange_integral_prime(m,k,kappa,i);
  return (trapez_sum+fb+fa);
}

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
  VectorXd unitcell_point(no_of_unitcell_pts);
  for(int i=0; i<unitcell_point.size(); i++) unitcell_point(i) = i*dx;
  for(int i=0; i<lattice_point.size(); i++) lattice_point(i) = i*dx;

  MatrixXcd H(no_of_unitcell_pts,no_of_unitcell_pts);
  spectrum eigenspectrum;
  VectorXcd v; MatrixXcd eigenvectors; VectorXd eigenvalues;

  for(int n=0; n<N; n++)
  {
    double k = (2*M_PI*n)/(N*a)-M_PI/a;
    for(int i=0; i<unitcell_point.size(); i++)
    {
      int j = (i==unitcell_point.size()-1)? 0 : i+1;
      H(i,j)= cd(-1/(2*dx*dx), -k/(2*dx));
      H(j,i)= cd(-1/(2*dx*dx), k/(2*dx));
      H(i,i)= cd(1/(dx*dx)+ pow(k,2)/2+ V(unitcell_point(i)), 0);
    }

    diagonalize(H,v,eigenvectors);

    for(int i=0; i<unitcell_point.size(); i++)  eigenspectrum.push_back(make_pair(v[i].real(), eigenvectors.col(i)));
    sort(eigenspectrum.begin(),eigenspectrum.end(),compare);
    arr[n] = eigenspectrum;
    eigenspectrum.clear();
  }

  // double k1,k2; cin >> k1 >> k2;
  // cout << direct_integral(k1,k2) << endl;
  // cout << exchange_integral(k1,k2) << endl;

  VectorXd correction= VectorXd::Zero(N);
  for(int i=0; i<N; i++)
  {
    for(int j=0; j<N; j++)
    {
      if(i==j) continue;
      double vd = 1/(4*M_PI)*direct_integral(0,i,j);  cd ve = 1/(4*M_PI)*exchange_integral(0,i,j);
      correction(i) += 2*vd-ve.real();
    }
    cout << "k= " << i << " done" << endl << endl;
  }

  for(int i=0; i<N; i++)
  {
    dataout << (2*M_PI*i)/(N*a)-M_PI/a << " " << arr[i].at(0).first << " " << arr[i].at(0).first+correction(i) << endl;
  }

  return 0;
}


// for(int n=0; n<N; n++)
// {
//   dataout << (2*M_PI*n)/(N*a)-M_PI/a << " ";
//   for(int i=0; i<5; i++) dataout << arr[n].at(i).first << " ";
//   dataout << endl;
// }
