#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double pi_sequential(long long n) {
  double sum = 0.0;

  for (long long i = 0; i < n; i++) {
    double term = 1.0 / (2.0 * i + 1.0);
    sum += (i % 2 == 0) ? term : -term;
  }

  return 4.0 * sum;
}

double pi_parallel(long long n) {
  double sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (long long i = 0; i < n; i++) {
    double term = 1.0 / (2.0 * i + 1.0);
    sum += (i % 2 == 0) ? term : -term;
  }

  return 4.0 * sum;
}

int main(int argc, char *argv[]) {
  long long n = 1000000000;

  if (argc > 1) {
    omp_set_num_threads(atoi(argv[1]));
  }

  if (argc > 2) {
    n = (long long) strtod(argv[2], NULL);
  }

  printf("n = %lld\n\n", n);
  int threads = omp_get_max_threads();

  double t = omp_get_wtime();
  double pi_seq = pi_sequential(n);
  double t_seq = omp_get_wtime() - t;

  printf("Последовательная версия:\n");
  printf("Вычисленное значение pi = %.15f\n", pi_seq);
  printf("Абсолютная ошибка = %.3e\n", fabs(pi_seq - M_PI));
  printf("Время выполнения = %.4f с\n", t_seq);
  printf("Количество потоков = 1\n\n");

  t = omp_get_wtime();
  double pi_par = pi_parallel(n);
  double t_par = omp_get_wtime() - t;

  printf("Параллельная версия:\n");
  printf("Вычисленное значение pi = %.15f\n", pi_par);
  printf("Абсолютная ошибка = %.3e\n", fabs(pi_par - M_PI));
  printf("Время выполнения = %.4f с\n", t_par);
  printf("Количество потоков = %d\n\n", threads);

  printf("Ускорение: %.2f\n", t_seq / t_par);
  return 0;
}
