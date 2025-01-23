#define MATRIX_IMPLEMENTATION
#define ALLOC_IMPLEMENTATION
#define RAND_IMPLEMENTATION

#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

static RandomSeries series;

int main(void)
{
    Arena* arena = ArenaMalloc(Kilobytes(1));
    series = RandomSeed(1020);

    Matrix m = MatrixAlloc(arena, 4, 4);
    Matrix w = MatrixAlloc(arena, 4, 4);

    MatrixFill(w, 1);
    MatrixPrint(w);

    MatrixRandomize(&series, m, -10, 10);
    MatrixPrint(m);

    MatrixSigmoid(m);
    MatrixPrint(m);

    ArenaFree(arena);
    return 0;
}
