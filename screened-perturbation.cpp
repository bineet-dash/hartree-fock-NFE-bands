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
int N = 20; //no of unit cells
int no_of_total_pts = no_of_unitcell_pts*N;
double init_pt= 0.0;
double final_pt = 10.0;
double A=1;
double epsilon = dx/2;
double electron_density = 2/a;
double lambda,Ef;

spectrum* arr = new spectrum [N];
VectorXd unitcell_point;
ofstream dataout("data_perturb_screened.txt");

double V(double x){return -A*(1+cos(2*M_PI*x/a));}
bool compare(const pair<double, VectorXcd>&i, const pair<double, VectorXcd>&j) {return i.first < j.first;}
cd u(int m, int k, int i) { return arr[k].at(m).second(i%no_of_unitcell_pts);}

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

double direct_core_integrand(int m, int k, int kappa, int i, int i_prime)
{
  return 1/(4*M_PI)*exp(-lambda*abs(unitcell_point(i)-unitcell_point(i_prime)))*norm(u(m,k,i))*norm(u(m,kappa,i_prime))/(abs(unitcell_point(i)-unitcell_point(i_prime))+epsilon);
}

double direct_integral(int m, int k, int kappa)
{
  double trapez_sum=0.0;
  for(int i=0; i< unitcell_point.size(); i++)
  {
    for(int i_prime=0; i_prime < unitcell_point.size(); i_prime++)
      trapez_sum+= direct_core_integrand(m,k,kappa,i,i_prime);
  }
  return trapez_sum;
}

cd exchange_core_integrand(int m, int k, int kappa, int i, int i_prime)
{
  cd num = conj(u(m,kappa,i))*u(m,kappa,i_prime)*conj(u(m,k,i_prime))*u(m,k,i)*exp(cd(0,(k-kappa)*(unitcell_point(i)-unitcell_point(i_prime))));
  cd denom = (abs(unitcell_point(i)-unitcell_point(i_prime))+epsilon);
  return 1/(4*M_PI)*exp(-lambda*abs(unitcell_point(i)-unitcell_point(i_prime)))*num/denom;
}

cd exchange_integral(int m, int k, int kappa)
{
  cd trapez_sum=cd(0,0);
  for(int i=0; i< unitcell_point.size(); i++)
  {
    for(int i_prime=0; i_prime<unitcell_point.size(); i_prime++)
      trapez_sum+= exchange_core_integrand(m,k,kappa,i,i_prime);
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

  Ef = arr[0].at(0).first; //Fermi Energy
  lambda = sqrt(3*electron_density/(2*Ef));
  cout << "lambda = " << lambda << endl;

  MatrixXd correction= MatrixXd::Zero(N,3);

  for(int m=0; m<3; m++)
  {
    double vd; cd ve;
    for(int i=0; i<N; i++)
    {
      for(int j=0; j<N; j++)
      {
        if(i==j) continue;
        vd = direct_integral(m,i,j); ve = exchange_integral(m,i,j);
        correction(i,m) += 2*vd-ve.real();
      }
    }
  }

  for(int i=0; i<N; i++)
  {
    dataout << (2*M_PI*i)/(N*a)-M_PI/a << " ";
    for(int m=0; m<3; m++) dataout << arr[i].at(m).first << " " << arr[i].at(m).first+correction(i,m) << " ";
    dataout << endl;
  }

  return 0;
}
