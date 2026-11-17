#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

typedef struct {
    int *data;
    int size;
    int capacity;
} ThreadList;

void init_list(ThreadList *list, int initial_cap) {
    list->capacity = initial_cap;
    list->size = 0;
    list->data = (int *)malloc(initial_cap * sizeof(int));
}

void append_list(ThreadList *list, int val) {
    if (list->size >= list->capacity) {
        list->capacity *= 2;
        list->data = (int *)realloc(list->data, list->capacity * sizeof(int));
    }
    list->data[list->size++] = val;
}

void free_list(ThreadList *list) {
    free(list->data);
}

void sequential_bfs(int n, int *offsets, int *edges, int start, int *dist, int *out_levels) {
    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[start] = 0;

    int *queue = (int *)malloc(n * sizeof(int));
    int head = 0, tail = 0;
    queue[tail++] = start;

    int max_level = 0;

    while (head < tail) {
        int u = queue[head++];
        if (dist[u] > max_level) {
            max_level = dist[u];
        }
        for (int j = offsets[u]; j < offsets[u + 1]; j++) {
            int v = edges[j];
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                queue[tail++] = v;
            }
        }
    }
    *out_levels = max_level + 1;
    free(queue);
}

static const char *schedule_name(omp_sched_t kind) {
    switch ((int)kind & 0x7fffffff) {
        case omp_sched_static:  return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided:  return "guided";
        case omp_sched_auto:    return "auto";
        default:                return "other";
    }
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) {
            printf("Использование: mpiexec -n <процессы> ./bfs <файл_графа> <начальная_вершина> [directed]\n");
        }
        MPI_Finalize();
        return 0;
    }

    char *filename = argv[1];
    int start_vertex = atoi(argv[2]);
    int directed = 0;
    if (argc >= 4 && (strcmp(argv[3], "directed") == 0 || strcmp(argv[3], "1") == 0)) {
        directed = 1;
    }

    if (getenv("OMP_SCHEDULE") == NULL) {
        omp_set_schedule(omp_sched_dynamic, 1);
    }

    int n = 0, m = 0;
    int *offsets = NULL;
    int *edges = NULL;

    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Ошибка: не удалось открыть файл %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (fscanf(f, "%d %d", &n, &m) != 2 || n <= 0 || m < 0) {
        fprintf(stderr, "Ошибка чтения N и M (ожидается N > 0, M >= 0)\n");
        fclose(f);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (start_vertex < 0 || start_vertex >= n) {
        if (rank == 0) {
            fprintf(stderr, "Ошибка: начальная вершина %d вне диапазона [0, %d]\n", start_vertex, n - 1);
        }
        fclose(f);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *temp_u = (int *)malloc((m > 0 ? m : 1) * sizeof(int));
    int *temp_v = (int *)malloc((m > 0 ? m : 1) * sizeof(int));
    int *degrees = (int *)calloc(n, sizeof(int));

    for (int i = 0; i < m; i++) {
        if (fscanf(f, "%d %d", &temp_u[i], &temp_v[i]) != 2) {
            fprintf(stderr, "Ошибка: не удалось прочитать ребро №%d\n", i);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (temp_u[i] < 0 || temp_u[i] >= n || temp_v[i] < 0 || temp_v[i] >= n) {
            fprintf(stderr, "Ошибка: ребро №%d (%d %d) содержит вершину вне диапазона [0, %d]\n",
                    i, temp_u[i], temp_v[i], n - 1);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        degrees[temp_u[i]]++;
        if (!directed) degrees[temp_v[i]]++;
    }
    fclose(f);

    offsets = (int *)malloc((n + 1) * sizeof(int));
    offsets[0] = 0;
    for (int i = 0; i < n; i++) {
        offsets[i + 1] = offsets[i] + degrees[i];
    }

    int total_edges = offsets[n];
    edges = (int *)malloc((total_edges > 0 ? total_edges : 1) * sizeof(int));
    int *current_offsets = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        current_offsets[i] = offsets[i];
    }

    for (int i = 0; i < m; i++) {
        int u = temp_u[i];
        int v = temp_v[i];
        edges[current_offsets[u]++] = v;
        if (!directed) edges[current_offsets[v]++] = u;
    }

    free(temp_u);
    free(temp_v);
    free(degrees);
    free(current_offsets);

    int *dist_seq = (int *)malloc(n * sizeof(int));
    int seq_levels = 0;
    double t_seq = 0.0;
    if (rank == 0) {
        double t0 = MPI_Wtime();
        sequential_bfs(n, offsets, edges, start_vertex, dist_seq, &seq_levels);
        t_seq = MPI_Wtime() - t0;
    }
    MPI_Bcast(dist_seq, n, MPI_INT, 0, MPI_COMM_WORLD);

    int *dist_par = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) dist_par[i] = -1;
    dist_par[start_vertex] = 0;

    int *frontier = (int *)malloc(n * sizeof(int));
    int *next_frontier = (int *)malloc(n * sizeof(int));
    int frontier_size = 1;
    frontier[0] = start_vertex;

    int *local_candidate = (int *)calloc(n, sizeof(int));
    int *global_candidate = (int *)calloc(n, sizeof(int));

    int max_threads = omp_get_max_threads();
    ThreadList *thread_candidates = (ThreadList *)malloc(max_threads * sizeof(ThreadList));
    for (int t = 0; t < max_threads; t++) init_list(&thread_candidates[t], 1024);

    int *frontier_sizes = (int *)malloc((n + 1) * sizeof(int));
    long long local_processed = 0;
    long long local_edges_scanned = 0;
    double comm_time = 0.0;

    int levels = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    while (frontier_size > 0) {
        frontier_sizes[levels] = frontier_size;
        levels++;

        memset(local_candidate, 0, n * sizeof(int));
        for (int t = 0; t < max_threads; t++) thread_candidates[t].size = 0;

        int local_frontier_size = (rank < frontier_size) ? (frontier_size - rank + nprocs - 1) / nprocs : 0;
        local_processed += local_frontier_size;

        long long scanned = 0;

        #pragma omp parallel reduction(+:scanned)
        {
            int tid = omp_get_thread_num();
            ThreadList *my_list = &thread_candidates[tid];

            #pragma omp for schedule(runtime)
            for (int k = 0; k < local_frontier_size; k++) {
                int idx = rank + k * nprocs;
                int v = frontier[idx];
                for (int j = offsets[v]; j < offsets[v + 1]; j++) {
                    int u = edges[j];
                    scanned++;
                    if (dist_par[u] == -1) {
                        append_list(my_list, u);
                    }
                }
            }
        }
        local_edges_scanned += scanned;

        for (int t = 0; t < max_threads; t++) {
            ThreadList *tl = &thread_candidates[t];
            for (int i = 0; i < tl->size; i++) {
                local_candidate[tl->data[i]] = 1;
            }
        }

        double c0 = MPI_Wtime();
        MPI_Allreduce(local_candidate, global_candidate, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - c0;

        int next_size = 0;
        for (int u = 0; u < n; u++) {
            if (global_candidate[u] == 1 && dist_par[u] == -1) {
                dist_par[u] = levels;
                next_frontier[next_size++] = u;
            }
        }

        int *temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;
        frontier_size = next_size;
    }

    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    double max_elapsed = 0.0, max_comm = 0.0;

    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long *all_processed = NULL, *all_scanned = NULL;
    if (rank == 0) {
        all_processed = (long long *)malloc(nprocs * sizeof(long long));
        all_scanned = (long long *)malloc(nprocs * sizeof(long long));
    }
    MPI_Gather(&local_processed, 1, MPI_LONG_LONG, all_processed, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_edges_scanned, 1, MPI_LONG_LONG, all_scanned, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int mismatches = 0;
    int unreachable_par = 0;
    for (int i = 0; i < n; i++) {
        if (dist_par[i] != dist_seq[i]) {
            mismatches++;
        }
        if (dist_par[i] == -1) {
            unreachable_par++;
        }
    }

    int max_mismatches = 0;
    MPI_Reduce(&mismatches, &max_mismatches, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int threads = omp_get_max_threads();
        long long total_threads = (long long)nprocs * threads;
        int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
        omp_sched_t sched_kind;
        int sched_chunk;
        omp_get_schedule(&sched_kind, &sched_chunk);

        double speedup = (max_elapsed > 0.0) ? t_seq / max_elapsed : 0.0;
        double efficiency = speedup / (double)total_threads;

        int max_front = 0;
        for (int l = 0; l < levels; l++) {
            if (frontier_sizes[l] > max_front) max_front = frontier_sizes[l];
        }

        printf("Граф: %s\n", directed ? "ориентированный" : "неориентированный");
        printf("Вершин: %d\n", n);
        printf("Рёбер: %d\n", m);
        printf("Средняя степень: %.2f\n", (double)total_edges / n);
        printf("Начальная вершина: %d\n", start_vertex);
        printf("Уровней: %d\n", levels);
        printf("Недостижимых вершин: %d\n", unreachable_par);
        printf("Несовпадающих расстояний: %d\n", max_mismatches);
        printf("Процессов: %d\n", nprocs);
        printf("Потоков на процесс: %d\n", threads);
        printf("Всего потоков: %lld\n", total_threads);
        printf("Планировка: %s,%d\n", schedule_name(sched_kind), sched_chunk);
        if (total_threads > cores) {
            printf("Потоков больше, чем ядер (%d)\n", cores);
        }
        printf("Время последовательное: %.6f\n", t_seq);
        printf("Время параллельное: %.6f\n", max_elapsed);
        printf("Время MPI_Allreduce: %.6f\n", max_comm);
        printf("Ускорение: %.3f\n", speedup);
        printf("Эффективность: %.3f\n", efficiency);
        printf("Максимальный фронт: %d\n", max_front);
        int limit = levels <= 50 ? levels : 50;
        for (int l = 0; l < limit; l++) {
            printf("Фронт %d: %d\n", l, frontier_sizes[l]);
        }
        long long max_proc = 0, sum_proc = 0;
        for (int p = 0; p < nprocs; p++) {
            printf("Процесс %d: вершин %lld, рёбер %lld\n", p, all_processed[p], all_scanned[p]);
            if (all_scanned[p] > max_proc) max_proc = all_scanned[p];
            sum_proc += all_scanned[p];
        }
        if (sum_proc > 0) {
            printf("Дисбаланс: %.3f\n", (double)max_proc * nprocs / sum_proc);
        }
        printf("Проверка: %s\n", max_mismatches == 0 ? "успешно" : "ошибка");

        free(all_processed);
        free(all_scanned);
    }

    for (int t = 0; t < max_threads; t++) free_list(&thread_candidates[t]);
    free(thread_candidates);
    free(frontier_sizes);
    free(offsets);
    free(edges);
    free(dist_seq);
    free(dist_par);
    free(frontier);
    free(next_frontier);
    free(local_candidate);
    free(global_candidate);

    MPI_Finalize();
    return 0;
}