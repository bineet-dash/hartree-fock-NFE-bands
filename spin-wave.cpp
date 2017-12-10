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

int no_of_unitcell_pts = 50;
double a = 1;
double dx = a/double(no_of_unitcell_pts);
int N = 20; //no of unit cells
double A=1;
int nu=2;

VectorXd unitcell_point(no_of_unitcell_pts);
VectorXd lattice_point(no_of_unitcell_pts*N);
VectorXd k_point(N);
spectrum* arr = new spectrum [N];

double V(double x){return -A*(1+cos(2*M_PI*x/a));}
bool compare(const pair<double, VectorXcd>&i, const pair<double, VectorXcd>&j) {return i.first < j.first;}
cd u(int m, int k, int i) { return arr[k].at(m).second(i%no_of_unitcell_pts);}
double E(int m, int k) {return arr[k].at(m).first;}

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

  return INFO==0;
}

VectorXd sortascending(VectorXd v1)
{
  vector<double> stdv1 (v1.data(),v1.data()+v1.size());
  std::sort (stdv1.begin(), stdv1.end());
  Map<ArrayXd> sorted(stdv1.data(), stdv1.size());
  return sorted;
}

cd W(int m, int Rn, int i)
{
  cd sum = 0;
  for(int k=0; k<k_point.size(); k++) sum+= u(m,k,i)*exp(cd(0,k_point(k)*(lattice_point(i)-Rn*a)));
  return 1/sqrt(k_point.size())*sum;
}

cd exchange_core_integrand(int m, int Rn1, int Rn2, int i, int i_prime)
{
  cd num = conj(W(m,Rn2,i))*W(m,Rn2,i_prime)*conj(W(m,Rn1,i_prime))*W(m,Rn1,i);
  cd denom = (abs(lattice_point(i)-lattice_point(i_prime))+dx/2);
  return 1/(4*M_PI)*num/denom;
}

cd exchange_integral(int m, int Rn1, int Rn2)
{
  cd trapez_sum=cd(0,0);
  for(int i=0; i< lattice_point.size(); i++)
  {
    for(int i_prime=0; i_prime<lattice_point.size(); i_prime++)
      trapez_sum+= exchange_core_integrand(m,Rn1,Rn2,i,i_prime);
    cout << i << " done." << "\r";
  }
  return trapez_sum;
}

double gamma(int k)
{
  return 2*cos(k_point(k)*a)/nu;
}

int main()
{
  MatrixXcd H(no_of_unitcell_pts,no_of_unitcell_pts);

  for(int i=0; i<unitcell_point.size(); i++) unitcell_point(i) = i*dx;
  for(int i=0; i<lattice_point.size(); i++) lattice_point(i) = i*dx;
  for(int i=0; i<N; i++) k_point(i) = (2*M_PI*i)/(N*a)-M_PI/a;

  vector < pair<double,VectorXcd> > eigenspectrum;
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
    eigenspectrum.resize(N);
    arr[n] = eigenspectrum;
    eigenspectrum.clear();
  }

  // ofstream debugout("check_bloch.txt");
  // for(int i=0; i<unitcell_point.size(); i++)
  // {
  //   int k=0;
  //   debugout << unitcell_point(i) << " ";
  //   for(int m=0; m<k_point.size(); m++) debugout << norm(u(m,k,i)) << " ";
  //   debugout << endl;
  // }

  // ofstream debugout("check_bloch.txt");
  // for(int i=0; i<lattice_point.size(); i++)
  // {
  //   int m=0;
  //   debugout << lattice_point(i) << " ";
  //   for(int Rn=0; Rn<N; Rn++) debugout << norm(W(m,Rn,i)) << " ";
  //   debugout << endl;
  // }

  double J0 = exchange_integral(0,2,3).real();
  double J1 = exchange_integral(1,2,3).real();
  double s = 1;

  //Ground State
  double E0_0 = -J0 * pow(s,2) * nu * N;
  double E0_1= -J1 * pow(s,2) * nu * N;

  cout << J0 << " " << J1 << endl;
  ofstream dataout("spin_wave.txt");
  for(int k=0; k<k_point.size(); k++)
    dataout << k_point(k) << " " << E0_0 + 2 * J0 * s * (1 - gamma(k)) << " " << E0_1 + 2 * J1 * s * (1 - gamma(k)) << endl;

  return 0;
}
