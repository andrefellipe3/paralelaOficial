Instruções para executar o código KmeansOMP.c (Código OpenMP para GPU)
Realizar o comando para compilar: gcc KmeansOMP.c -o omp -fopenmp -O3 -lm
Para executar, use o comando: time ./omp

O código OpenMP que esta realizando a paralelização levando os dados para a GPU é o "KmeansOMP.c", os outros arquivos como paral.c e seq.c é referente ao código sequencial e ao código paralelo sem levar pra GPU.

OBS: Antes de compilar o KmeansOMP, extraia a base de dados que esta nomeada como "bd.zip".
