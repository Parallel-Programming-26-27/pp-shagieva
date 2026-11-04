#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define RUNS 5

void transpose_rows(const double *A, double *B, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      B[j * N + i] = A[i * N + j];
    }
  }
}

void transpose_cols(const double *A, double *B, int N) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      B[j * N + i] = A[i * N + j];
    }
  }
}

void transpose_blocked(const double *A, double *B, int N, int b) {
  for (int ii = 0; ii < N; ii += b) {
    for (int jj = 0; jj < N; jj += b) {
      int i_end = (ii + b < N) ? ii + b : N;
      int j_end = (jj + b < N) ? jj + b : N;

      for (int i = ii; i < i_end; i++) {
        for (int j = jj; j < j_end; j++) {
          B[j * N + i] = A[i * N + j];
        }
      }
    }
  }
}

void transpose_rows_omp(const double *A, double *B, int N, int chunk) {
  if (chunk == 0) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        B[j * N + i] = A[i * N + j];
      }
    }
  } else {
    #pragma omp parallel for schedule(static, chunk)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        B[j * N + i] = A[i * N + j];
      }
    }
  }
}

void transpose_blocked_omp(const double *A, double *B, int N, int b) {
  #pragma omp parallel for schedule(static)
  for (int ii = 0; ii < N; ii += b) {
    for (int jj = 0; jj < N; jj += b) {
      int i_end = (ii + b < N) ? ii + b : N;
      int j_end = (jj + b < N) ? jj + b : N;

      for (int i = ii; i < i_end; i++) {
        for (int j = jj; j < j_end; j++) {
          B[j * N + i] = A[i * N + j];
        }
      }
    }
  }
}

void transpose_inplace(double *A, int N, int b) {
  for (int ii = 0; ii < N; ii += b) {
    for (int jj = ii; jj < N; jj += b) {
      int i_end = (ii + b < N) ? ii + b : N;
      int j_end = (jj + b < N) ? jj + b : N;

      for (int i = ii; i < i_end; i++) {
        int j_start = (ii == jj) ? i + 1 : jj;
        for (int j = j_start; j < j_end; j++) {
          double tmp = A[i * N + j];
          A[i * N + j] = A[j * N + i];
          A[j * N + i] = tmp;
        }
      }
    }
  }
}

void transpose_inplace_omp(double *A, int N, int b) {
  #pragma omp parallel for schedule(dynamic, 1)
  for (int ii = 0; ii < N; ii += b) {
    for (int jj = ii; jj < N; jj += b) {
      int i_end = (ii + b < N) ? ii + b : N;
      int j_end = (jj + b < N) ? jj + b : N;

      for (int i = ii; i < i_end; i++) {
        int j_start = (ii == jj) ? i + 1 : jj;
        for (int j = j_start; j < j_end; j++) {
          double tmp = A[i * N + j];
          A[i * N + j] = A[j * N + i];
          A[j * N + i] = tmp;
        }
      }
    }
  }
}

void run_version(const char *version, const double *A, double *B, int N, int b, int chunk) {
  if (strcmp(version, "rows") == 0) {
    transpose_rows(A, B, N);
  } else if (strcmp(version, "cols") == 0) {
    transpose_cols(A, B, N);
  } else if (strcmp(version, "blocked") == 0) {
    transpose_blocked(A, B, N, b);
  } else if (strcmp(version, "rows_omp") == 0) {
    transpose_rows_omp(A, B, N, chunk);
  } else if (strcmp(version, "blocked_omp") == 0) {
    transpose_blocked_omp(A, B, N, b);
  } else if (strcmp(version, "inplace") == 0) {
    transpose_inplace(B, N, b);
  } else if (strcmp(version, "inplace_omp") == 0) {
    transpose_inplace_omp(B, N, b);
  }
}

void prepare(const char *version, const double *A, double *B, int N) {
  size_t bytes = (size_t) N * N * sizeof(double);

  if (strncmp(version, "inplace", 7) == 0) {
    memcpy(B, A, bytes);
  } else {
    memset(B, 0, bytes);
  }
}

long long check(const double *B, const double *ref, int N) {
  long long errors = 0;

  for (long long k = 0; k < (long long) N * N; k++) {
    if (B[k] != ref[k]) {
      errors++;
    }
  }

  return errors;
}

int compare_double(const void *x, const void *y) {
  double a = *(const double *) x;
  double b = *(const double *) y;
  return (a > b) - (a < b);
}

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int b = atoi(argv[2]);
  int threads = atoi(argv[3]);
  const char *version = argv[4];
  int chunk = (argc > 5) ? atoi(argv[5]) : 0;

  omp_set_num_threads(threads);

  size_t bytes = (size_t) N * N * sizeof(double);
  double *A = malloc(bytes);
  double *B = malloc(bytes);
  double *ref = malloc(bytes);

  for (long long k = 0; k < (long long) N * N; k++) {
    A[k] = (double) k;
  }

  transpose_rows(A, ref, N);

  prepare(version, A, B, N);
  run_version(version, A, B, N, b, chunk);

  double times[RUNS];
  long long errors = 0;

  for (int r = 0; r < RUNS; r++) {
    prepare(version, A, B, N);

    double t = omp_get_wtime();
    run_version(version, A, B, N, b, chunk);
    times[r] = omp_get_wtime() - t;

    errors += check(B, ref, N);
  }

  qsort(times, RUNS, sizeof(double), compare_double);
  double median = times[RUNS / 2];
  double bw = 2.0 * N * N * sizeof(double) / median / 1e9;

  printf("N=%d b=%d threads=%d version=%s", N, b, threads, version);
  if (argc > 5) {
    printf(" schedule=static,%d", chunk);
  }
  printf(" time=%.6f s BW=%.2f GB/s check=%s\n", median, bw, errors == 0 ? "OK" : "FAIL");

  free(A);
  free(B);
  free(ref);
  return 0;
}