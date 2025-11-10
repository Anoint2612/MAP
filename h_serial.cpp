#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
using namespace std;
using namespace std::complex_literals;

int main() {
    int N = 1024; // lattice size
    int steps = 1000;
    double dt = 0.01;
    double J = 1.0;

    vector<complex<double>> psi(N, 1.0);
    
    auto start = chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < N - 1; i++) {
            // simple 2-site Trotter update (toy example)
            complex<double> temp = psi[i];
            psi[i] = cos(J*dt)*psi[i] - 1i*sin(J*dt)*psi[i+1];
            psi[i+1] = cos(J*dt)*psi[i+1] - 1i*sin(J*dt)*temp;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Serial runtime: "
         << chrono::duration<double>(end - start).count()
         << " s" << endl;
}
