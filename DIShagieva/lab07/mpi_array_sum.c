#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int rank, size;
    long long N = -1;
    long long reps = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc >= 2)
            N = atoll(argv[1]);
        if (argc >= 3)
            reps = atoll(argv[2]);
    }

    MPI_Bcast(&N, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reps, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (N <= 0) {
        MPI_Finalize();
        return 1;
    }

    int divisible = (N % size == 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    long long base = N / size, rem = N % size, off = 0;
    for (int i = 0; i < size; i++) {
        counts[i] = (int)(base + (i < rem ? 1 : 0));
        displs[i] = (int)off;
        off += counts[i];
    }
    int local_n = counts[rank];

    long long *a = NULL;
    if (rank == 0) {
        a = malloc(N * sizeof(long long));
        for (long long i = 0; i < N; i++)
            a[i] = i + 1;
    }

    long long *part = malloc((local_n > 0 ? local_n : 1) * sizeof(long long));
    if (divisible)
        MPI_Scatter(a, local_n, MPI_LONG_LONG_INT, part, local_n,
                    MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    else
        MPI_Scatterv(a, counts, displs, MPI_LONG_LONG_INT, part, local_n,
                     MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < local_n; i++)
        local_sum += part[i];

    long long *sums = NULL;
    if (rank == 0)
        sums = malloc(size * sizeof(long long));
    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, sums, 1, MPI_LONG_LONG_INT,
               0, MPI_COMM_WORLD);
    long long gather_result = 0;
    if (rank == 0)
        for (int i = 0; i < size; i++)
            gather_result += sums[i];

    long long reduce_result = 0;
    MPI_Reduce(&local_sum, &reduce_result, 1, MPI_LONG_LONG_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    long long allreduce_result = 0;
    MPI_Allreduce(&local_sum, &allreduce_result, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    double t_local = MPI_Wtime() - t0, t_max;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long expected = N * (N + 1) / 2;
    int ok_local = (allreduce_result == expected), ok_all;
    MPI_Reduce(&ok_local, &ok_all, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("P = %d, N = %lld (%s)\n", size, N,
               divisible ? "MPI_Scatter" : "MPI_Scatterv");
        printf("Ожидаемая сумма: %lld\n", expected);
        printf("Gather: %lld %s\n", gather_result,
               gather_result == expected ? "OK" : "ОШИБКА");
        printf("Reduce: %lld %s\n", reduce_result,
               reduce_result == expected ? "OK" : "ОШИБКА");
        printf("Allreduce: %lld %s (у всех процессов: %s)\n", allreduce_result,
               allreduce_result == expected ? "OK" : "ОШИБКА",
               ok_all ? "совпадает" : "не совпадает");
        printf("Время выполнения: %.6f с\n", t_max);
    }

    if (reps > 0) {
        long long r_g = 0, r_r = 0, tmp = 0;
        double tg, tr, tg_max, tr_max;

        MPI_Barrier(MPI_COMM_WORLD);
        tg = MPI_Wtime();
        for (long long k = 0; k < reps; k++) {
            MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, sums, 1,
                       MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                tmp = 0;
                for (int i = 0; i < size; i++)
                    tmp += sums[i];
                r_g = tmp;
            }
        }
        tg = MPI_Wtime() - tg;

        MPI_Barrier(MPI_COMM_WORLD);
        tr = MPI_Wtime();
        for (long long k = 0; k < reps; k++)
            MPI_Reduce(&local_sum, &r_r, 1, MPI_LONG_LONG_INT, MPI_SUM,
                       0, MPI_COMM_WORLD);
        tr = MPI_Wtime() - tr;

        MPI_Reduce(&tg, &tg_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tr, &tr_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Повторений: %lld\n", reps);
            printf("Gather + сумма: %.9f с/операция, результат %s\n",
                   tg_max / reps, r_g == expected ? "корректен" : "ОШИБКА");
            printf("Reduce: %.9f с/операция, результат %s\n",
                   tr_max / reps, r_r == expected ? "корректен" : "ОШИБКА");
        }
    }

    free(counts);
    free(displs);
    free(part);
    free(a);
    free(sums);
    MPI_Finalize();
    return 0;
}