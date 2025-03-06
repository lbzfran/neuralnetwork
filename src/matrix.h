/*
 * ---------------
 * Liam Bagabag
 * Version: 2.0.0
 * dependencies: alloc.h (specific), random.h (specific)
 * requires: MATRIX_IMPLEMENTATION
 * ---------------
 */
#ifndef MATRIX_H
#define MATRIX_H

// NOTE(liam): replace stdint include with base
#include <stdio.h>
#include <math.h>

#include "arena.h"
#include "random.h"

#ifndef m_alloc
# include <stdlib.h>
# define m_alloc malloc
# define m_realloc realloc
# define m_free free
#endif

// NOTE(liam): specific math functions for neural net operations
// TODO(liam): likely move these in nn.h once its ready.
float reluf(float x);
float sigmoidf(float x);
/*float softmaxf(float x);*/

typedef struct Matrix {
    size_t rows;
    size_t cols;
    float *V;
} Matrix, Row;

#define MatrixAT(m, i, j) ((m).V[(i) * m.cols + (j)])
#define RowAT(r, i) MatrixAT(r, 0, i)

Matrix MatrixAlloc(size_t, size_t, float*);

#define MatrixArenaAlloc(arena, i, j) (MatrixAlloc((i), (j), PushArray(arena, float, (i) * (j))))
#define RowArenaAlloc(arena, i) MatrixArenaAlloc(arena, 1, i)

Matrix MatrixMalloc(size_t, size_t);
void MatrixFree(Matrix a);

void MatrixPrint_(Matrix a, const char*);
#define MatrixPrint(m) MatrixPrint_(m, #m)

void MatrixRandomize(RandomSeries *, Matrix, float, float);
void MatrixFill(Matrix, float);

Row MatrixRow(Matrix, size_t);
void MatrixCopy(Matrix, Matrix);

void MatrixAddScalar(Matrix, Matrix, float);
void MatrixAddMatrix(Matrix, Matrix, Matrix);
void MatrixSubScalar(Matrix, Matrix, float);
void MatrixSubMatrix(Matrix, Matrix, Matrix);
void MatrixDot(Matrix, Matrix, Matrix);
void MatrixMulMatrix(Matrix, Matrix, Matrix); // Hadamard Product

void MatrixSum(Matrix, Matrix);
void MatrixTranspose(Matrix, Matrix);

void MatrixShuffleValue(RandomSeries *, Matrix);

bool32 MatrixRandomShuffleRow(RandomSeries *, Matrix, size_t, size_t *);
bool32 MatrixRandomShuffleCol(RandomSeries *, Matrix, size_t, size_t *);
bool32 MatrixShuffleRow(Matrix, size_t *, size_t);
bool32 MatrixShuffleCol(Matrix, size_t *, size_t);
#define RowRandomShuffle MatrixRandomShuffleCol
#define RowShuffle MatrixShuffleCol

void MatrixApply(Matrix, float (*)(float));
void MatrixSigmoid(Matrix); // "apply" wrapper with sigmoidf

#endif

#ifdef MATRIX_IMPLEMENTATION

float
sigmoidf(float x)
{
    // NOTE(liam): return value between 0 and 1 based on how far along
    // the given float is from -inf to inf.
    return 1.f / (1.f + expf(-x));
}

float
reluf(float x)
{
    return x ? x > 0 : 0;
}

/*float*/
/*softmax(float x)*/
/*{*/
/**/
/*}*/

Matrix
MatrixAlloc(size_t rows, size_t cols, float* V)
{
    Matrix res;

    res.rows = rows;
    res.cols = cols;
    res.V = V;

    return res;
}

Matrix
MatrixMalloc(size_t rows, size_t cols)
{
    Matrix res;

    res.rows = rows;
    res.cols = cols;
    res.V = (float*)m_alloc(sizeof(float) * rows * cols);
    Assert(!res.V && "Malloc failed during Matrix Allocation.");

    return res;
}

void
MatrixFree(Matrix a)
{
    if (a.V)
    {
        m_free(a.V);
    }
}

void
MatrixPrint_(Matrix a, const char* name)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < a.cols; j++)
        {
            printf("\t%f ", MatrixAT(a, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

Matrix
MatrixRow(Matrix a, size_t row)
{
    return (Row) {
        .rows = 1,
        .cols = a.cols,
        .V = &MatrixAT(a, row, 0),
    };
}

void
MatrixRandomize(RandomSeries* series, Matrix a, float low, float high)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            MatrixAT(a, i, j) = RandomBetween(series, low, high);
        }
    }
}

void
MatrixCopy(Matrix b, Matrix a)
{
    Assert(a.rows == b.rows);
    Assert(a.cols == b.cols);
    for (size_t i = 0; i < b.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            MatrixAT(b, i, j) = MatrixAT(a, i, j);
        }
    }

}

void
MatrixFill(Matrix a, float x)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            MatrixAT(a, i, j) = x;
        }
    }
}

void
MatrixAddScalar(Matrix b, Matrix a, float x)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            /*b->V[i][j] = a->V[i][j] + x;*/
            MatrixAT(b, i, j) = MatrixAT(a, i, j) + x;
        }
    }
}

void
MatrixAddMatrix(Matrix c, Matrix a, Matrix b)
{
    Assert(a.rows == b.rows);
    Assert(a.cols == b.cols);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            /*c->V[i][j] = a->V[i][j] + b->V[i][j];*/
            /*c.V[i * c.cols + j] = a.V[i * a.cols + j] + b.V[i * b.cols
             * + j];*/
            MatrixAT(c, i, j) =
                MatrixAT(a, i, j) + MatrixAT(b, i, j);
        }
    }
}

void
MatrixSum(Matrix b, Matrix a)
{
    Assert(b.rows == a.rows);
    Assert(b.cols == a.cols);
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            MatrixAT(b, i, j) += MatrixAT(a, i, j);
        }
    }
}

void
MatrixSubScalar(Matrix b, Matrix a, float x)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            /*b->V[i][j] = a->V[i][j] + x;*/
            MatrixAT(b, i, j) = MatrixAT(a, i, j) - x;
        }
    }
}

void
MatrixSubMatrix(Matrix c, Matrix a, Matrix b)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            /*c->V[i][j] = a->V[i][j] - b->V[i][j];*/
            MatrixAT(c, i, j) = MatrixAT(a, i, j) - MatrixAT(b, i, j);
        }
    }
}

void
MatrixDot(Matrix c, Matrix a, Matrix b)
{
    // NOTE(liam): implied that both matrix dimensions are the same.
    Assert(a.cols == b.rows);
    Assert(a.rows == c.rows);
    Assert(c.cols == b.cols);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            MatrixAT(c, i, j) = 0;
            for (size_t k = 0; k < b.rows; k++) {
                MatrixAT(c, i, j) += MatrixAT(a, i, k) * MatrixAT(b, k, j);
            }
        }
    }
}

void
MatrixMulMatrix(Matrix c, Matrix a, Matrix b)
{
    Assert(a.rows == b.rows);
    Assert(c.rows == b.rows);
    Assert(c.cols == a.cols);
    Assert(b.cols == a.cols);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            MatrixAT(c, i, j) = MatrixAT(a, i, j) * MatrixAT(b, i, j);
        }
    }
}

void
MatrixApply(Matrix a, float (*fun)(float))
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            a.V[i * a.cols + j] = (*fun)(a.V[i * a.cols + j]);
        }
    }
}

void
MatrixTranspose(Matrix b, Matrix a)
{
    for (size_t i = 0; i < b.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            /*b.V[i * b.cols + j] = a.V[j * a.cols + i];*/
            MatrixAT(b, i, j) = MatrixAT(a, j, i);
        }
    }
}

void
MatrixSigmoid(Matrix a)
{
    MatrixApply(a, sigmoidf);
}

void
MatrixReLU(Matrix a)
{
    MatrixApply(a, reluf);
}

void
MatrixShuffleValue(RandomSeries *series, Matrix a)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            uint32 row = RandomChoice(series, a.rows);
            uint32 col = RandomChoice(series, a.cols);

            float32 tmp = MatrixAT(a, i, j);
            MatrixAT(a, i, j) = MatrixAT(a, row, col);
            MatrixAT(a, row, col) = tmp;
        }
    }
}

bool32
MatrixRandomShuffleRow(RandomSeries *series, Matrix a, size_t count_per_pair, size_t *shuffle_indices)
{
    // NOTE(liam) shuffle_indices' size must be at least a.cols
    bool32 res = true;
    if (series == NULL)
    {
        res = false;
    }
    else
    {
        size_t *pos = shuffle_indices;

        while (count_per_pair--)
        {
            size_t rowA = RandomChoice(series, a.rows);
            size_t rowB = RandomChoice(series, a.rows);

            if (rowA == rowB) continue;

            if (pos)
            {
                *(pos++) = rowA;
                *(pos++) = rowB;
            }

            for (size_t j = 0; j < a.cols; j++)
            {
                float32 tmp = MatrixAT(a, rowA, j);
                MatrixAT(a, rowA, j) = MatrixAT(a, rowB, j);
                MatrixAT(a, rowB, j) = tmp;
            }
        }
    }
    return res;
}

bool32
MatrixRandomShuffleCol(RandomSeries * series, Matrix a, size_t count_per_pair, size_t *shuffle_indices)
{
    // NOTE(liam): shuffle_indices can be nullable if user does not need to
    // know the shuffle indices. indices are inserted 2 per count_per_pair: (from, to)
    // therefore assume the following: shuffle_indices[count_per_pair * 2]
    bool32 res = true;
    if (series == NULL)
    {
        res = false;
    }
    else
    {
        size_t *pos = shuffle_indices;

        while (count_per_pair--)
        {
            size_t colA = RandomChoice(series, a.cols);
            size_t colB = RandomChoice(series, a.cols);

            if (colA == colB) continue;

            if (pos)
            {
                *(pos++) = colA;
                *(pos++) = colB;
            }

            for (size_t i = 0; i < a.rows; i++)
            {
                float32 tmp = MatrixAT(a, i, colA);
                MatrixAT(a, i, colA) = MatrixAT(a, i, colB);
                MatrixAT(a, i, colB) = tmp;
            }
        }
    }
    return res;
}

bool32
MatrixShuffleRow(Matrix a, size_t *shuffle_indices, size_t count_per_pair)
{
    // NOTE(liam): count_per_pair == sizeof(shuffle_indices) / 2
    bool32 res = true;
    if (shuffle_indices == NULL)
    {
        res = false;
    }
    else
    {
        size_t pos = 0;
        while (count_per_pair-- > 0)
        {
            // TODO(liam): i feel like this operation is not efficiently indexed.
            size_t rowA = shuffle_indices[pos++];
            size_t rowB = shuffle_indices[pos++];

            Assert(rowA < a.rows);
            Assert(rowB < a.rows);
            if (rowA == rowB) continue;

            for (size_t j = 0; j < a.cols; j++)
            {
                float32 tmp = MatrixAT(a, rowA, j);
                MatrixAT(a, rowA, j) = MatrixAT(a, rowB, j);
                MatrixAT(a, rowB, j) = tmp;
            }
        }
    }
    return res;
}

bool32
MatrixShuffleCol(Matrix a, size_t *shuffle_indices, size_t count_per_pair)
{
    bool32 res = true;
    if (shuffle_indices == NULL)
    {
        res = false;
    }
    else
    {
        size_t pos = 0;
        while (count_per_pair-- > 0)
        {
            size_t colA = shuffle_indices[pos++];
            size_t colB = shuffle_indices[pos++];
            /*printf("DEBUG: shuffling col %lu -> %lu.\n", colA, colB);*/

            Assert(colA < a.cols);
            Assert(colB < a.cols);
            if (colA == colB) continue;

            for (size_t i = 0; i < a.rows; i++)
            {
                float32 tmp = MatrixAT(a, i, colA);
                MatrixAT(a, i, colA) = MatrixAT(a, i, colB);
                MatrixAT(a, i, colB) = tmp;
            }
        }
    }
    return res;
}

#endif
