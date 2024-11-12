#include<iostream>
#include<math.h>
#include<time.h>
#include "hadamard.cuh"

#define SIZE pow(2, 20)
#define BLOCK_SIZE 1024


int main()
{
    struct timespec start, stop; 
    double time;
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

    float *a_h, *b_h, *dest_h;
    float *a_d, *b_d, *dest_d;

    a_h = (float *)malloc(sizeof(float) * SIZE);
    b_h = (float *)malloc(sizeof(float) * SIZE);
    dest_h = (float *)malloc(sizeof(float) * SIZE);

    cudaMalloc((void **)&a_d, SIZE * sizeof(float));
    cudaMalloc((void **)&b_d, SIZE * sizeof(float));
    cudaMalloc((void **)&dest_d, SIZE * sizeof(float));


    for (int i = 0; i < SIZE; i++) {
        a_h[i] = (float)rand() / RAND_MAX;
        b_h[i] = (float)rand() / RAND_MAX;
        dest_h[i] = 0;
    }

    cudaMemcpy(a_d, a_h, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    multiply_them<<<grid, block>>>(dest_d, a_d, b_d);

    cudaMemcpy(dest_h, dest_d, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i< 5; i++)
    {
        printf("%f ", dest_h[i] - (a_h[i] * b_h[i]));
    }

    free(a_h);
    free(b_h);
    free(dest_h);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(dest_d);

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("\nTime is %f ms\n", time*1e3);

    return 0;
}