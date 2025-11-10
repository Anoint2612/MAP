#include <mpi.h>
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::complex_literals;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024;
    int local_N = N / size;
    double dt = 0.01;
    double J = 1.0;
    int steps = 1000;

    vector<complex<double>> psi(local_N, 1.0);
    complex<double> left, right;

    auto start = MPI_Wtime();

    for (int s = 0; s < steps; s++) {
        // Exchange boundary values
        if (rank > 0)
            MPI_Sendrecv(&psi[0], 1, MPI_CXX_DOUBLE_COMPLEX, rank - 1, 0,
                         &left, 1, MPI_CXX_DOUBLE_COMPLEX, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank < size - 1)
            MPI_Sendrecv(&psi[local_N - 1], 1, MPI_CXX_DOUBLE_COMPLEX, rank + 1, 0,
                         &right, 1, MPI_CXX_DOUBLE_COMPLEX, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Local updates
        for (int i = 0; i < local_N - 1; i++) {
            complex<double> temp = psi[i];
            psi[i] = cos(J*dt)*psi[i] - 1i*sin(J*dt)*psi[i+1];
            psi[i+1] = cos(J*dt)*psi[i+1] - 1i*sin(J*dt)*temp;
        }
    }

    auto end = MPI_Wtime();
    double runtime = end - start;

    double max_runtime;
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "Parallel runtime (" << size << " procs): " << max_runtime << " s" << endl;

    MPI_Finalize();
}
