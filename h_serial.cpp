#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <fstream>
#include <unistd.h>
using namespace std;
using namespace std::complex_literals;



int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./h_serial <N>\n";
        return 1;
    }

    int N = stoi(argv[1]);
    int steps = 1000;
    double dt = 0.01, J = 1.0;

    vector<complex<double>> psi(N, 1.0);
    double psi_mb = (N * sizeof(complex<double>)) / (1024.0 * 1024.0);

    auto t0 = chrono::high_resolution_clock::now();

    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < N - 1; i++) {
            complex<double> temp = psi[i];
            psi[i] = cos(J*dt)*psi[i] - 1i*sin(J*dt)*psi[i+1];
            psi[i+1] = cos(J*dt)*psi[i+1] - 1i*sin(J*dt)*temp;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(t1 - t0).count();

    cout << "Serial runtime: " << runtime << " s\n";

    return 0;
}
