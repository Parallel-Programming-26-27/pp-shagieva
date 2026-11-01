#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_MESSAGE 1
#define TAG_REPLY 2
#define MAX_LEN 64

void base_exchange(int rank) {
  char msg[MAX_LEN], reply[MAX_LEN];
  MPI_Status status;

  if (rank == 0) {
    strcpy(msg, "Hello from process 0");
    MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
    printf("[rank 0] sent to rank 1: \"%s\"\n", msg);

    MPI_Recv(reply, MAX_LEN, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, &status);
    printf("[rank 0] received from rank %d: \"%s\"\n", status.MPI_SOURCE, reply);
  } else if (rank == 1) {
    MPI_Recv(msg, MAX_LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
    printf("[rank 1] received from rank %d: \"%s\"\n", status.MPI_SOURCE, msg);

    strcpy(reply, "Message received");
    MPI_Send(reply, strlen(reply) + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    printf("[rank 1] sent reply to rank 0\n");
  }
}

void all_exchange(int rank, int size) {
  char msg[MAX_LEN], reply[MAX_LEN];
  MPI_Status status;

  if (rank == 0) {
    for (int r = 1; r < size; r++) {
      snprintf(msg, MAX_LEN, "Hello, process %d", r);
      MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, r, TAG_MESSAGE, MPI_COMM_WORLD);
      printf("[rank 0] sent to rank %d: \"%s\"\n", r, msg);
    }
    for (int r = 1; r < size; r++) {
      MPI_Recv(reply, MAX_LEN, MPI_CHAR, r, TAG_REPLY, MPI_COMM_WORLD, &status);
      printf("[rank 0] received from rank %d: \"%s\"\n", status.MPI_SOURCE, reply);
    }
  } else {
    MPI_Recv(msg, MAX_LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, &status);
    printf("[rank %d] received from rank %d: \"%s\"\n", rank, status.MPI_SOURCE, msg);

    snprintf(reply, MAX_LEN, "Process %d: message received", rank);
    MPI_Send(reply, strlen(reply) + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
  }
}

void ping_pong(int rank, char *buf, int n, int count) {
  for (int k = 0; k < count; k++) {
    if (rank == 0) {
      MPI_Send(buf, n, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
      MPI_Recv(buf, n, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv(buf, n, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(buf, n, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
  }
}

void timing(int rank) {
  const int sizes[] = {1, 1024, 65536, 1048576};
  const int reps = 1000;
  char *buf = calloc(1048576, 1);

  for (int s = 0; s < 4; s++) {
    ping_pong(rank, buf, sizes[s], 10);

    double t0 = MPI_Wtime();
    ping_pong(rank, buf, sizes[s], reps);
    double avg = (MPI_Wtime() - t0) / reps;

    if (rank == 0) printf("size=%d bytes reps=%d avg_time=%.3e s\n", sizes[s], reps, avg);
  }

  free(buf);
}

int main(int argc, char **argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    if (rank == 0) printf("Error: at least 2 processes are required\n");
    MPI_Finalize();
    return 1;
  }

  const char *mode = (argc > 1) ? argv[1] : "base";

  if (strcmp(mode, "all") == 0) {
    all_exchange(rank, size);
  } else if (strcmp(mode, "time") == 0) {
    timing(rank);
  } else {
    base_exchange(rank);
  }

  MPI_Finalize();
  return 0;
}
