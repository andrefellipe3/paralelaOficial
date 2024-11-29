#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_POINTS 253681  // Ajuste para o número total de pontos na base de dados
#define NUM_DIMENSIONS 21   // Número de variáveis ou colunas numéricas
#define K 5
#define MAX_ITERATIONS 100
#define NUM_THREADS 1 //AQUI VOCÊ MUDA O NUMERO DE THREADS
#define NUM_TEAMS 2 

double euclidean_distance(double *a, double *b, int dimensions) {
    double distance = 0.0;
    for (int i = 0; i < dimensions; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

void kmeans_parallel(double points[NUM_POINTS][NUM_DIMENSIONS], int labels[NUM_POINTS], double centroids[K][NUM_DIMENSIONS]) {
    int iterations = 0;
    while (iterations < MAX_ITERATIONS) {
        int changes = 0;
        
        
        #pragma omp target teams num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) \
            map(to: points[0:NUM_POINTS][0:NUM_DIMENSIONS], centroids[0:K][0:NUM_DIMENSIONS]) \
            map(tofrom: labels[0:NUM_POINTS]) \
            reduction(+: changes)
        #pragma omp distribute parallel for

        for (int i = 0; i < NUM_POINTS; i++) 
        {
            int nearest_centroid = 0;
            double min_distance = euclidean_distance(points[i], centroids[0], NUM_DIMENSIONS);

            for (int j = 1; j < K; j++) 
            {
                double distance = euclidean_distance(points[i], centroids[j], NUM_DIMENSIONS);
                if (distance < min_distance) 
                {
                    min_distance = distance;
                    nearest_centroid = j;
                }
            }

            if (labels[i] != nearest_centroid)
             {
                labels[i] = nearest_centroid;
                changes++;
            }
        }

        double new_centroids[K][NUM_DIMENSIONS] = {0};
        int counts[K] = {0};

        omp_set_num_threads(NUM_THREADS);
         #pragma omp parallel
        {
            double local_centroids[K][NUM_DIMENSIONS] = {0};
            int local_counts[K] = {0};

            omp_set_num_threads(NUM_THREADS);
            #pragma omp for
            for (int i = 0; i < NUM_POINTS; i++) {
                int cluster = labels[i];
                local_counts[cluster]++;
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    local_centroids[cluster][d] += points[i][d];
                }
            }

            omp_set_num_threads(NUM_THREADS);
            #pragma omp critical
            {
                for (int j = 0; j < K; j++) {
                    counts[j] += local_counts[j];
                    for (int d = 0; d < NUM_DIMENSIONS; d++) {
                        new_centroids[j][d] += local_centroids[j][d];
                    }
                }
            }
        }

        for (int j = 0; j < K; j++) {
            if (counts[j] > 0) {
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    centroids[j][d] = new_centroids[j][d] / counts[j];
                }
            }
        }

        if (changes == 0) {
            break;
        }

        iterations++;
    }
}

int main() {
    double (*points)[NUM_DIMENSIONS] = malloc(NUM_POINTS * sizeof(*points));
    int *labels = malloc(NUM_POINTS * sizeof(*labels));
    double centroids[K][NUM_DIMENSIONS];

    FILE *file = fopen("processed_data_diabetes.csv", "r");
    if (!file) {
        printf("Erro ao abrir o arquivo.\n");
        return 1;
    }

    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < NUM_DIMENSIONS; j++) {
            fscanf(file, "%lf,", &points[i][j]);
        }
        labels[i] = 0;
    }
    fclose(file);

    for (int j = 0; j < K; j++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
            centroids[j][d] = rand() % 100; 
        }
    }

    double start = omp_get_wtime();
    kmeans_parallel(points, labels, centroids);
    double end = omp_get_wtime();

    printf("Tempo de execução: %f segundos\n", end - start);

    free(points);
    free(labels);

    return 0;
}
