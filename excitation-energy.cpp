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

int no_of_unitcell_pts=20;
double a = 1;
double dx = a/double(no_of_unitcell_pts);
int N = 10; //no of unit cells
int no_of_total_pts = no_of_unitcell_pts*N;
double init_pt= 0.0;
double final_pt = 10.0;
double A=1;
double epsilon = dx/2;

spectrum* arr = new spectrum [N];
VectorXd unitcell_point;

double V(double x){return -A*(1+cos(2*M_PI*x/a));}
bool compare(const pair<double, VectorXcd>&i, const pair<double, VectorXcd>&j) {return i.first < j.first;}
cd u(int m, int k, int i) { return arr[k].at(m).second(i%no_of_unitcell_pts);}
double E(int m, int k) {return arr[k].at(m).first;}
cd psi(int m, int k, int i) { return u(m,k,i)*unitcell_point(i);}

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
  int LWORK = int(A.size())*10;
  VectorXcd WORK(LWORK);
  VectorXd RWORK(5*LDA);

  zgeev_(&Nchar, &Vchar, &N, reinterpret_cast <__complex__ double*> (A.data()), &LDA, reinterpret_cast <__complex__ double*> (w), 0, &LDV, reinterpret_cast <__complex__ double*> (v.data()), &LDV,  reinterpret_cast <__complex__ double*> (WORK.data()), &LWORK, RWORK.data(), &INFO);

  return INFO==0;
}

cd needed_exchange_integrand(int m, int n, int k, int i, int i_prime)
{
  cd num = conj(psi(n,k,i))*conj(u(m,k,i_prime))*u(m,k,i_prime)*u(m,k,i);
  cd denom = (abs(unitcell_point(i)-unitcell_point(i_prime))+epsilon);
  return 1/(4*M_PI)*num/denom;
}

cd needed_exchange(int m, int n, int k)
{
  cd trapez_sum=cd(0,0);
  for(int i=0; i< unitcell_point.size(); i++)
  {
    for(int i_prime=0; i_prime<unitcell_point.size(); i_prime++)
      trapez_sum+= needed_exchange_integrand(m,n,k,i,i_prime);
  }
  return trapez_sum;
}

double needed_direct_integrand(int m, int n, int k, int i, int i_prime)
{
  return 1/(4*M_PI)*norm(u(m,k,i))*norm(u(n,k,i_prime))/(abs(unitcell_point(i)-unitcell_point(i_prime))+epsilon);
}

double needed_direct(int m, int n, int k)
{
  double trapez_sum=0.0;
  for(int i=0; i< unitcell_point.size(); i++)
  {
    for(int i_prime=0; i_prime < unitcell_point.size(); i_prime++)
      trapez_sum+= needed_direct_integrand(m,n, k,i,i_prime);
  }
  return trapez_sum;
}



int main()
{
  unitcell_point.resize(no_of_unitcell_pts);
  for(int i=0; i<unitcell_point.size(); i++) unitcell_point(i) = i*dx;

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

  double DeltaW = E(1,0)-E(0,0);
  cout << "Original DeltaW= " << DeltaW << endl;
  DeltaW = DeltaW - needed_direct(0,1,0) + 2*needed_exchange(0,1,0).real();
  cout << "Corrected DeltaW= " << DeltaW << endl;


  return 0;
}
