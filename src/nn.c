
#define MATRIX_IMPLEMENTATION
#define ALLOC_IMPLEMENTATION
#define RAND_IMPLEMENTATION

#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

static RandomSeries local_series;

typedef struct {
    Matrix a0;
    Matrix w1, b1, a1;
    Matrix w2, b2, a2;
} Xor;

float forward_xor(Xor m)
{
    MatrixDot(m.a1, m.a0, m.w1);
    MatrixSum(m.a1, m.b1);
    MatrixSigmoid(m.a1);

    MatrixDot(m.a2, m.a1, m.w2);
    MatrixSum(m.a2, m.b2);
    MatrixSigmoid(m.a2);

    return *m.a2.V;
}

float cost(Xor m, Matrix ti, Matrix to)
{
    Assert(ti.rows == to.rows, "");
    Assert(to.cols == m.a2.cols, "");
    size_t n = ti.rows;

    float c = 0;
    for (size_t i = 0; i < n; i++) {
        Matrix x = MatrixRow(ti, i);
        Matrix y = MatrixRow(to, i);

        MatrixCopy(m.a0, x);
        forward_xor(m);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++) {
            float d = MatrixAT(m.a2, 0, j) - MatrixAT(y, 0, j);
            c += d * d;
        }

    }
    return c/n;
}

float ti[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

int main(void)
{
    Arena* local_arena = ArenaMalloc(Kilobytes(2));
    local_series = RandomSeed(122);

    Xor m;

    Matrix ti = MatrixArenaAlloc(local_arena, 2, 1);
    Matrix to = MatrixArenaAlloc(local_arena, 2, 1);

    m.a0 = MatrixArenaAlloc(local_arena, 1, 2);

    m.w1 = MatrixArenaAlloc(local_arena, 2, 2);
    m.b1 = MatrixArenaAlloc(local_arena, 1, 2);
    m.a1 = MatrixArenaAlloc(local_arena, 1, 2); // activation

    m.w2 = MatrixArenaAlloc(local_arena, 2, 1);
    m.b2 = MatrixArenaAlloc(local_arena, 1, 1);
    m.a2 = MatrixArenaAlloc(local_arena, 1, 1);

    MatrixRandomize(&local_series, m.w1, 0, 1);
    MatrixRandomize(&local_series, m.w2, 0, 1);
    MatrixRandomize(&local_series, m.b1, 0, 1);
    MatrixRandomize(&local_series, m.b2, 0, 1);

    printf("cost = %f\n", cost(m, ti, to));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MatrixAT(m.a0, 0, 0) = i;
            MatrixAT(m.a0, 0, 1) = j;
            forward_xor(m);
            float y = *m.a2.V;

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }

    ArenaFree(local_arena);
    return 0;
}
