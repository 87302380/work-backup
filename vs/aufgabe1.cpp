#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int n = 12;
    if (argv[1] == std::string("s")) {
        n = atoi(argv[2]);
    }


    int myid, size, source;
    MPI_Status status;
    int i;
    int message_length = 4;
    int send[n];
    int recv[n];


    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0) {
        for (i = 0; i < n; i++) {
            send[i] = 1;
        }
    }
    for (i = 0; i < n; i++) {
            recv[i] = 0;
        }

    printf("Running with array of size %d on %d processes \n", n, size);

    //Calculate Size of Send
    int sendcounts[size];
    int displs[size];
    int pos = 0;
    int rem = n % size;

    //printf("N/Size = %d, rest = %d \n", n/size, rem);
    for (int i = 0; i < size; i++) {
        sendcounts[i] = n/size;
        if (rem > 0){
            sendcounts[i]++;
            rem--;
        }
        displs[i] = pos;
        pos += sendcounts[i];
    }

    MPI_Scatterv(send, sendcounts, displs, MPI_INT, recv, n, MPI_INT, 0, MPI_COMM_WORLD);
    int sum = 0;
    for (i = 0; i < n; i++) {
        sum = sum + recv[i];
        printf("myid = %d, recv[%d] = %d\n", myid, i, recv[i]);
    }
    printf("myid is %d, my sum is %d\n", myid, sum);

    int sumback[size];
    MPI_Gather(&sum, 1, MPI_INT, sumback, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (myid == 0) {
        int result = 0;
        for (i = 0; i < size; i++) {
            result = result + sumback[i];
            printf("received the sum result from process %d is %d\n", i, sumback[i]);
        }
        printf("The sum of all returned results is %d\n", result);

    }


    MPI_Finalize();
    return 0;
}
