/*
* Формат запуска: OMP_NUM_THREADS=p ./matmul N b [variant]
 * N - размер матриц, b - размер блока, p - число потоков,
 * variant - ijk_seq, ijk_omp, ikj_seq, ikj_omp или blocked_omp (без него - все пять).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define RUNS 3
#define TOL 1e-9

void ijk_seq(const double *A, const double *B, double *C, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

void ijk_omp(const double *A, const double *B, double *C, int N) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

void ikj_seq(const double *A, const double *B, double *C, int N) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      double a = A[i * N + k];
      for (int j = 0; j < N; j++) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}

void ikj_omp(const double *A, const double *B, double *C, int N) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      double a = A[i * N + k];
      for (int j = 0; j < N; j++) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}

void blocked_omp(const double *A, const double *B, double *C, int N, int b) {
  #pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < N; ii += b) {
    for (int jj = 0; jj < N; jj += b) {
      int i_end = (ii + b < N) ? ii + b : N;
      int j_end = (jj + b < N) ? jj + b : N;

      for (int kk = 0; kk < N; kk += b) {
        int k_end = (kk + b < N) ? kk + b : N;

        for (int i = ii; i < i_end; i++) {
          for (int k = kk; k < k_end; k++) {
            double a = A[i * N + k];
            for (int j = jj; j < j_end; j++) {
              C[i * N + j] += a * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

void run_variant(const char *name, const double *A, const double *B, double *C, int N, int b) {
  if (strcmp(name, "ijk_seq") == 0) {
    ijk_seq(A, B, C, N);
  } else if (strcmp(name, "ijk_omp") == 0) {
    ijk_omp(A, B, C, N);
  } else if (strcmp(name, "ikj_seq") == 0) {
    ikj_seq(A, B, C, N);
  } else if (strcmp(name, "ikj_omp") == 0) {
    ikj_omp(A, B, C, N);
  } else if (strcmp(name, "blocked_omp") == 0) {
    blocked_omp(A, B, C, N, b);
  }
}

long long check(const double *C, const double *ref, int N, double *max_err) {
  long long errors = 0;
  *max_err = 0.0;

  for (long long k = 0; k < (long long) N * N; k++) {
    double diff = C[k] - ref[k];
    if (diff < 0) diff = -diff;
    double scale = ref[k] < 0 ? -ref[k] : ref[k];
    if (scale < 1.0) scale = 1.0;

    if (diff > *max_err) *max_err = diff;
    if (diff > TOL * scale) errors++;
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

  const char *all[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};
  const char **variants = all;
  int count = 5;
  if (argc > 3) {
    variants = (const char **) &argv[3];
    count = 1;
  }

  int threads = omp_get_max_threads();
  size_t bytes = (size_t) N * N * sizeof(double);

  double *A = malloc(bytes);
  double *B = malloc(bytes);
  double *C = malloc(bytes);
  double *ref = malloc(bytes);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = (double) ((i * 3 + j * 7) % 10) / 10.0;
      B[i * N + j] = (double) ((i * 5 + j * 2) % 10) / 10.0;
    }
  }

  memset(ref, 0, bytes);
  ijk_seq(A, B, ref, N);

  printf("N=%d b=%d threads=%d runs=%d\n", N, b, threads, RUNS);

  for (int v = 0; v < count; v++) {
    const char *name = variants[v];
    double times[RUNS];
    long long errors = 0;
    double max_err = 0.0;

    for (int r = 0; r < RUNS; r++) {
      memset(C, 0, bytes);

      double t = omp_get_wtime();
      run_variant(name, A, B, C, N, b);
      times[r] = omp_get_wtime() - t;

      double err;
      errors += check(C, ref, N, &err);
      if (err > max_err) max_err = err;
    }

    qsort(times, RUNS, sizeof(double), compare_double);
    double median = times[RUNS / 2];
    int t_used = strstr(name, "_seq") ? 1 : threads;

    printf("variant=%-11s threads=%-2d time=%.6f s check=%s (max_err=%.1e)\n",
           name, t_used, median, errors == 0 ? "OK" : "FAIL", max_err);
  }

  free(A);
  free(B);
  free(C);
  free(ref);
  return 0;
}