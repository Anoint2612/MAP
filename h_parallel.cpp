#include <mpi.h>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <unistd.h>
using namespace std;
using namespace std::complex_literals;

// peak RSS (KB)
size_t get_peak_rss_kb() {
    long size, resident;
    ifstream statm("/proc/self/statm");
    if (!statm.is_open()) return 0;
    statm >> size >> resident;
    long page_kb = sysconf(_SC_PAGESIZE) / 1024;
    return resident * page_kb;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: ./h_parallel <N>\n";
        MPI_Finalize();
        return 1;
    }

    int N = stoi(argv[1]);
    int steps = 1000;
    double dt = 0.01, J = 1.0;

    // Domain decomposition
    int base = N / size;
    int rem = N % size;
    int local_N = base + (rank < rem ? 1 : 0);

    vector<complex<double>> psi(local_N, 1.0);
    double psi_mb = (local_N * sizeof(complex<double>)) / (1024.0 * 1024.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int s = 0; s < steps; s++) {
        complex<double> left_recv = 0, right_recv = 0;
        complex<double> left_send = psi[0], right_send = psi[local_N - 1];

        int left_rank  = (rank == 0 ? MPI_PROC_NULL : rank - 1);
        int right_rank = (rank == size-1 ? MPI_PROC_NULL : rank + 1);

        MPI_Sendrecv(&left_send, 1, MPI_CXX_DOUBLE_COMPLEX, left_rank, 0,
                     &right_recv, 1, MPI_CXX_DOUBLE_COMPLEX, right_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&right_send, 1, MPI_CXX_DOUBLE_COMPLEX, right_rank, 1,
                     &left_recv, 1, MPI_CXX_DOUBLE_COMPLEX, left_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // apply boundaries
        if (left_rank != MPI_PROC_NULL)      psi[0] = left_recv;
        if (right_rank != MPI_PROC_NULL)     psi[local_N - 1] = right_recv;

        // local update
        for (int i = 0; i < local_N - 1; i++) {
            complex<double> temp = psi[i];
            psi[i] = cos(J*dt)*psi[i] - 1i*sin(J*dt)*psi[i+1];
            psi[i+1] = cos(J*dt)*psi[i+1] - 1i*sin(J*dt)*temp;
        }
    }

    double t1 = MPI_Wtime();
    double runtime = t1 - t0;

    MPI_Finalize();
    return 0;
}
