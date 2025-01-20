/*#include "matrix.h"*/

#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
    int rows = 4;
    int cols = 6;
    int* V = (int*)malloc(rows * cols * sizeof(int));
    for (int i = 0;
            i < rows;
            i++)
    {
        for (int j = 0;
                j < cols;
                j++)
        {
            V[i * cols + j] = (i + j) % 2;
        }
    }

    for (int i = 0;
            i < rows;
            i++)
    {
        for (int j = 0;
                j < cols;
                j++)
        {
            printf("%d ", V[i * cols + j]);
        }
        printf("\n");
    }

    free(V);
    return 0;
}
