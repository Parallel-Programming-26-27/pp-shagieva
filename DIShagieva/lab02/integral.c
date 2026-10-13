#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x) {
  return (x * x - 1.0) * pow(10.0, -2.0 * x);
}

double F(double x) {
  double k = 2.0 * log(10.0);
  return -exp(-k * x) * ((x * x - 1.0) / k + 2.0 * x / (k * k) + 2.0 / (k * k * k));
}

double integral_exact(double a, double b) {
  return F(b) - F(a);
}

double integral_sequential(double a, double b, long long n) {
  double h = (b - a) / n;
  double sum = 0.0;

  for (long long i = 0; i < n; i++) {
    sum += f(a + (i + 0.5) * h);
  }

  return sum * h;
}

double integral_parallel(double a, double b, long long n) {
  double h = (b - a) / n;
  double sum = 0.0;

  #pragma omp parallel for reduction(+:sum)
  for (long long i = 0; i < n; i++) {
    sum += f(a + (i + 0.5) * h);
  }

  return sum * h;
}

double integral_eps(double a, double b, double eps, int parallel, long long *n_out) {
  long long n = 10;
  double prev = parallel ? integral_parallel(a, b, n) : integral_sequential(a, b, n);
  double cur;

  while (1) {
    n *= 2;
    cur = parallel ? integral_parallel(a, b, n) : integral_sequential(a, b, n);
    if (fabs(cur - prev) < eps) {
      break;
    }
    prev = cur;
  }

  *n_out = n;
  return cur;
}

void print_result(const char *title, double value, double exact, double time, int threads) {
  printf("%s:\n", title);
  printf("Вычисленное значение интеграла = %.15f\n", value);
  printf("Абсолютная ошибка = %.3e\n", fabs(value - exact));
  printf("Время выполнения = %.6f с\n", time);
  printf("Количество потоков = %d\n\n", threads);
}

void experiment1(double a, double b, long long n) {
  int thread_counts[] = {1, 2, 4, 8, 16};
  double exact = integral_exact(a, b);

  printf("Эксперимент 1. Влияние количества потоков\n");
  printf("a = %g, b = %g, n = %lld\n\n", a, b, n);

  double t = omp_get_wtime();
  double seq = integral_sequential(a, b, n);
  double t_seq = omp_get_wtime() - t;

  print_result("Последовательная версия", seq, exact, t_seq, 1);

  for (int k = 0; k < 5; k++) {
    omp_set_num_threads(thread_counts[k]);

    t = omp_get_wtime();
    double par = integral_parallel(a, b, n);
    double t_par = omp_get_wtime() - t;

    print_result("Параллельная версия", par, exact, t_par, thread_counts[k]);
    printf("Ускорение: %.2f\n\n", t_seq / t_par);
  }
}

void experiment2(double a, double b, int threads) {
  double exact = integral_exact(a, b);
  omp_set_num_threads(threads);

  printf("Эксперимент 2. Влияние числа разбиений\n");
  printf("a = %g, b = %g\n\n", a, b);

  for (long long n = 10000; n <= 100000000LL; n *= 10) {
    double t = omp_get_wtime();
    double par = integral_parallel(a, b, n);
    double t_par = omp_get_wtime() - t;

    printf("n = %lld\n", n);
    print_result("Параллельная версия", par, exact, t_par, threads);
  }
}

void experiment3(double a, double b, double eps, int threads) {
  double exact = integral_exact(a, b);
  long long n_seq, n_par;
  omp_set_num_threads(threads);

  printf("Эксперимент 3. Вычисление с заданной точностью\n");
  printf("a = %g, b = %g, eps = %g\n", a, b, eps);
  printf("Условие остановки: |I_2n - I_n| < eps, начальное n = 10\n\n");

  double t = omp_get_wtime();
  double seq = integral_eps(a, b, eps, 0, &n_seq);
  double t_seq = omp_get_wtime() - t;

  printf("Найденное n = %lld\n", n_seq);
  print_result("Последовательная версия", seq, exact, t_seq, 1);

  t = omp_get_wtime();
  double par = integral_eps(a, b, eps, 1, &n_par);
  double t_par = omp_get_wtime() - t;

  printf("Найденное n = %lld\n", n_par);
  print_result("Параллельная версия", par, exact, t_par, threads);

  printf("Ускорение: %.2f\n", t_seq / t_par);
}

int main(int argc, char *argv[]) {

  int experiment = atoi(argv[1]);
  double a = atof(argv[2]);
  double b = atof(argv[3]);

  switch (experiment) {
    case 1:
      experiment1(a, b, (long long) strtod(argv[4], NULL));
      break;
    case 2:
      experiment2(a, b, atoi(argv[4]));
      break;
    case 3:
      experiment3(a, b, atof(argv[4]), atoi(argv[5]));
      break;
  }

  return 0;
}
