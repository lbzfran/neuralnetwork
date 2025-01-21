#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

void
MatrixPrint(Matrix a)
{
    for (int i = 0;
            i < a.rows;
            i++)
    {
        for (int j = 0;
                j < a.cols;
                j++)
        {
            printf("%d ", a.V[i * a.cols + j]);
        }
        printf("\n");
    }
    printf("dim: %lu %lu\n", a.rows, a.cols);
}

int
main(void)
{

    Arena* arena = ArenaMalloc(Kilobytes(1));

    Matrix matA = {
        .rows = 4,
        .cols = 6,
        .V = (int*)malloc(matA.rows * matA.cols * sizeof(int))
    };
    Matrix matB = {
        .rows = 6,
        .cols = 4,
        .V = (int*)malloc(matB.rows * matB.cols * sizeof(int))
    };

    for (int i = 0;
            i < matA.rows;
            i++)
    {
        for (int j = 0;
                j < matA.cols;
                j++)
        {
            matA.V[i * matA.cols + j] = (i + 1) * (j + 1);
        }
    }

    // 0 = row
    // 1 = col
    matA.V[ 0 * matA.cols + 1 ] = 2;
    printf("matA:\n");
    MatrixPrint(matA);
    printf("\n");
    printf("matB:\n");
    MatrixPrint(matB);
    printf("\n");
    /**/
    /*printf("perf mat add\n");*/
    /*MatrixAddMatrix(&matA, &matB, &matA);*/
    /**/
    /*MatrixPrint(matA);*/
    /*printf("\n");*/
    /**/
    /*MatrixAddScalar(&matA, &matA, 2);*/
    /**/
    /*MatrixPrint(matA);*/
    /*printf("\n");*/
    printf("Transpose:\n");

    MatrixTranspose(matA, &matB);

    MatrixPrint(matB);
    printf("\n");
    /*free(matA.V);*/
    /*free(matB.V);*/
    ArenaFree(arena);
    return 0;
}
