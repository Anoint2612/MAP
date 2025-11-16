// h_parallel_10k.cpp
#include <mpi.h>
#include <vector>
#include <complex>
#include <iostream>
using namespace std;
using namespace std::complex_literals;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank==0) cerr << "Usage: ./h_parallel_10k <N>\n";
        MPI_Finalize();
        return 1;
    }

    int N = stoi(argv[1]);
    const int steps = 10000;   // increased
    const double dt = 0.01;
    const double J = 1.0;

    // simple block distribution with remainder handled (but we'll pick N divisible by core count)
    int base = N / size;
    int rem  = N % size;
    int local_N = base + (rank < rem ? 1 : 0);

    vector<complex<double>> psi(local_N, 1.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int s = 0; s < steps; ++s) {
        complex<double> left_recv = 0.0, right_recv = 0.0;
        complex<double> left_send = psi[0];
        complex<double> right_send = psi[local_N - 1];

        int left_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        int right_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        // exchange boundary values
        MPI_Sendrecv(&left_send, 1, MPI_CXX_DOUBLE_COMPLEX, left_rank, 0,
                     &right_recv, 1, MPI_CXX_DOUBLE_COMPLEX, right_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&right_send, 1, MPI_CXX_DOUBLE_COMPLEX, right_rank, 1,
                     &left_recv, 1, MPI_CXX_DOUBLE_COMPLEX, left_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // apply boundaries (if exist)
        if (left_rank != MPI_PROC_NULL) psi[0] = left_recv;
        if (right_rank != MPI_PROC_NULL) psi[local_N - 1] = right_recv;

        // local updates
        for (int i = 0; i < local_N - 1; ++i) {
            complex<double> temp = psi[i];
            psi[i] = cos(J*dt)*psi[i] - 1i*sin(J*dt)*psi[i+1];
            psi[i+1] = cos(J*dt)*psi[i+1] - 1i*sin(J*dt)*temp;
        }
    }

    double t1 = MPI_Wtime();
    double runtime = t1 - t0;

    // Print per-rank timing line exactly how Python driver expects it.
    cout << "Rank " << rank << " | time = " << runtime << " s\n";

    MPI_Finalize();
    return 0;
}
