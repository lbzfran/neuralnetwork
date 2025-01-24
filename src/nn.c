
#define MATRIX_IMPLEMENTATION
#define ALLOC_IMPLEMENTATION
#define RAND_IMPLEMENTATION

#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

static RandomSeries local_series;

int main(void)
{
    Arena* local_arena = ArenaMalloc(Kilobytes(2));
    local_series = RandomSeed(122);

    Matrix x  = MatrixAlloc(local_arena, 1, 2);
    Matrix w1 = MatrixAlloc(local_arena, 2, 2);
    Matrix b1 = MatrixAlloc(local_arena, 1, 2);
    Matrix a1 = MatrixAlloc(local_arena, 1, 2); // activation

    Matrix w2 = MatrixAlloc(local_arena, 2, 1);
    Matrix b2 = MatrixAlloc(local_arena, 1, 1);
    Matrix a2 = MatrixAlloc(local_arena, 1, 1);

    MatrixRandomize(&local_series, w1, 0, 1);
    MatrixRandomize(&local_series, w2, 0, 1);
    MatrixRandomize(&local_series, b1, 0, 1);
    MatrixRandomize(&local_series, b2, 0, 1);

    float x1 = 0;
    float x2 = 1;
    MatrixAT(x, 0, 0) = x1;
    MatrixAT(x, 0, 1) = x2;

    MatrixDot(a1, x, w1);
    MatrixSum(a1, b1);
    MatrixSigmoid(a1);

    MatrixDot(a2, a1, w2);
    MatrixSum(a2, b2);
    MatrixSigmoid(a2);

    float y = *a2.V;

    MatrixPrint(x);
    MatrixPrint(a1);
    MatrixPrint(w1);
    MatrixPrint(b1);
    MatrixPrint(w2);
    MatrixPrint(b2);
    MatrixPrint(a2);

    ArenaFree(local_arena);
    return 0;
}
