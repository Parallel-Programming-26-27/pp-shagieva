/*
 * Формат запуска: OMP_NUM_THREADS=p ./linreg_simd [N] [d] [K] [lambda] [eta]
 *   N — число объектов (по умолчанию 100000)
 *   d — число признаков (по умолчанию 32)
 *   K — число итераций обучения (по умолчанию 100)
 *   lambda — коэффициент L1 (по умолчанию 0.001)
 *   eta — шаг обучения (по умолчанию 0.5)
 *   p — число потоков для OpenMP-версий
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define RUNS 3
#define TOL 1e-6
#define EPS 1e-6

void grad_serial(const double *X, const double *y, const double *w, double b,
                 int N, int d, double *g, double *gb) {
  double sb = 0.0;
  memset(g, 0, d * sizeof(double));

  for (int i = 0; i < N; i++) {
    const double *x = &X[(size_t) i * d];

    double pred = b;
    for (int j = 0; j < d; j++) pred += x[j] * w[j];
    double e = pred - y[i];

    sb += e;
    for (int j = 0; j < d; j++) g[j] += e * x[j];
  }

  for (int j = 0; j < d; j++) g[j] /= N;
  *gb = sb / N;
}

void grad_omp(const double *X, const double *y, const double *w, double b,
              int N, int d, double *g, double *gb) {
  double sb = 0.0;
  memset(g, 0, d * sizeof(double));

  /*
   * Объекты (строки X) делятся между потоками поровну непрерывными кусками.
   * reduction: у каждого потока свои копии sb и g, после цикла они складываются, поэтому гонок нет.
  */
  #pragma omp parallel for schedule(static) reduction(+: sb, g[:d])
  for (int i = 0; i < N; i++) {
    const double *x = &X[(size_t) i * d];

    double pred = b;
    for (int j = 0; j < d; j++) pred += x[j] * w[j];
    double e = pred - y[i];

    sb += e;
    for (int j = 0; j < d; j++) g[j] += e * x[j];
  }

  for (int j = 0; j < d; j++) g[j] /= N;
  *gb = sb / N;
}

void grad_simd(const double *X, const double *y, const double *w, double b,
               int N, int d, double *g, double *gb) {
  double sb = 0.0;
  memset(g, 0, d * sizeof(double));

  for (int i = 0; i < N; i++) {
    const double *x = &X[(size_t) i * d];

    double pred = b;
    /*
     * Векторизация: одна инструкция умножает сразу несколько пар x[j] * w[j].
     * reduction: каждая позиция вектора копит свою часть pred, в конце части складываются.
     */
    #pragma omp simd reduction(+: pred)
    for (int j = 0; j < d; j++) pred += x[j] * w[j];
    double e = pred - y[i];

    sb += e;
    /*
     * Векторизация: одна инструкция обновляет сразу несколько соседних g[j].
     * Каждая итерация пишет в свой g[j], итерации независимы, поэтому reduction не нужен.
    */
    #pragma omp simd
    for (int j = 0; j < d; j++) g[j] += e * x[j];
  }

  for (int j = 0; j < d; j++) g[j] /= N;
  *gb = sb / N;
}

void grad_omp_simd(const double *X, const double *y, const double *w, double b,
                   int N, int d, double *g, double *gb) {
  double sb = 0.0;
  memset(g, 0, d * sizeof(double));

  /*
   * Объекты (строки X) делятся между потоками поровну непрерывными кусками.
   * reduction: у каждого потока свои копии sb и g, после цикла они складываются, поэтому гонок нет.
  */
  #pragma omp parallel for schedule(static) reduction(+: sb, g[:d])
  for (int i = 0; i < N; i++) {
    const double *x = &X[(size_t) i * d];

    double pred = b;
    /*
     * Векторизация: одна инструкция умножает сразу несколько пар x[j] * w[j].
     * reduction: каждая позиция вектора копит свою часть pred, в конце части складываются.
    */
    #pragma omp simd reduction(+: pred)
    for (int j = 0; j < d; j++) pred += x[j] * w[j];
    double e = pred - y[i];

    sb += e;
    /*
     * Векторизация: одна инструкция обновляет сразу несколько соседних g[j].
     * Каждая итерация пишет в свой g[j], итерации независимы, поэтому reduction не нужен.
    */
    #pragma omp simd
    for (int j = 0; j < d; j++) g[j] += e * x[j];
  }

  for (int j = 0; j < d; j++) g[j] /= N;
  *gb = sb / N;
}

typedef void (*grad_fn)(const double *, const double *, const double *, double,
                        int, int, double *, double *);

double soft_threshold(double z, double a) {
  if (z > a) return z - a;
  if (z < -a) return z + a;
  return 0.0;
}

void train(grad_fn grad, const double *X, const double *y, int N, int d,
           int K, double lambda, double eta, double *w, double *b, double *g) {
  memset(w, 0, d * sizeof(double));
  *b = 0.0;

  for (int k = 0; k < K; k++) {
    double gb;
    grad(X, y, w, *b, N, d, g, &gb);

    for (int j = 0; j < d; j++) w[j] = soft_threshold(w[j] - eta * g[j], eta * lambda);
    *b -= eta * gb;
  }
}

double loss(const double *X, const double *y, const double *w, double b,
            int N, int d, double lambda) {
  double s = 0.0, l1 = 0.0;
  for (int i = 0; i < N; i++) {
    double pred = b;
    for (int j = 0; j < d; j++) pred += X[(size_t) i * d + j] * w[j];
    s += (pred - y[i]) * (pred - y[i]);
  }
  for (int j = 0; j < d; j++) l1 += fabs(w[j]);
  return s / (2.0 * N) + lambda * l1;
}

int compare_double(const void *x, const void *y) {
  double a = *(const double *) x, b = *(const double *) y;
  return (a > b) - (a < b);
}

int main(int argc, char *argv[]) {
  int N = argc > 1 ? atoi(argv[1]) : 100000;
  int d = argc > 2 ? atoi(argv[2]) : 32;
  int K = argc > 3 ? atoi(argv[3]) : 100;
  double lambda = argc > 4 ? atof(argv[4]) : 1e-3;
  double eta = argc > 5 ? atof(argv[5]) : 0.5;

  double *X = malloc((size_t) N * d * sizeof(double));
  double *y = malloc(N * sizeof(double));
  double *w_true = malloc(d * sizeof(double));
  double *w = malloc(d * sizeof(double));
  double *w_ref = malloc(d * sizeof(double));
  double *g = malloc(d * sizeof(double));

  srand(42);
  for (int j = 0; j < d; j++) w_true[j] = (j % 2 == 0) ? (double) (j + 1) / d : 0.0;
  for (int i = 0; i < N; i++) {
    double t = 0.5;
    for (int j = 0; j < d; j++) {
      X[(size_t) i * d + j] = 2.0 * rand() / RAND_MAX - 1.0;
      t += X[(size_t) i * d + j] * w_true[j];
    }
    y[i] = t + 0.1 * (2.0 * rand() / RAND_MAX - 1.0);
  }

  const char *names[] = {"serial", "omp", "simd", "omp_simd"};
  grad_fn fns[] = {grad_serial, grad_omp, grad_simd, grad_omp_simd};
  int threads = omp_get_max_threads();

  printf("N=%d d=%d K=%d lambda=%g eta=%g threads=%d\n", N, d, K, lambda, eta, threads);

  for (int v = 0; v < 4; v++) {
    double times[RUNS], b;

    for (int r = 0; r < RUNS; r++) {
      double t = omp_get_wtime();
      train(fns[v], X, y, N, d, K, lambda, eta, w, &b, g);
      times[r] = omp_get_wtime() - t;
    }
    qsort(times, RUNS, sizeof(double), compare_double);

    if (v == 0) memcpy(w_ref, w, d * sizeof(double));

    double dw = 0.0;
    int zeros = 0;
    for (int j = 0; j < d; j++) {
      if (fabs(w[j] - w_ref[j]) > dw) dw = fabs(w[j] - w_ref[j]);
      if (fabs(w[j]) < EPS) zeros++;
    }

    printf("variant=%-8s threads=%-2d time=%.6f s loss=%.10f dw=%.1e zeros=%d/%d check=%s\n",
           names[v], (v == 1 || v == 3) ? threads : 1, times[RUNS / 2],
           loss(X, y, w, b, N, d, lambda), dw, zeros, d, dw < TOL ? "OK" : "FAIL");
  }

  free(X); free(y); free(w_true); free(w); free(w_ref); free(g);
  return 0;
}