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
#define RowAT(r, i) MatrixAT(r, 1, i)

Matrix MatrixAlloc(size_t rows, size_t cols, float*);

#define MatrixArenaAlloc(arena, i, j) (MatrixAlloc((i), (j), PushArray(arena, float, (i) * (j))))
#define RowArenaAlloc(arena, i) MatrixArenaAlloc(arena, 1, i)

Matrix MatrixMalloc(size_t rows, size_t cols);
void MatrixFree(Matrix a);

void MatrixPrint_(Matrix a, const char* name);
#define MatrixPrint(m) MatrixPrint_(m, #m)

void MatrixRandomize(RandomSeries* series, Matrix a, float low, float high);
void MatrixFill(Matrix, float);

Matrix MatrixRow(Matrix, size_t);
void MatrixCopy(Matrix, Matrix);

void MatrixAddScalar(Matrix, Matrix, float);
void MatrixAddMatrix(Matrix, Matrix, Matrix);
void MatrixSubScalar(Matrix, Matrix, float);
void MatrixSubMatrix(Matrix, Matrix, Matrix);
void MatrixDot(Matrix, Matrix, Matrix);

void MatrixSum(Matrix, Matrix);
void MatrixTranspose(Matrix, Matrix);

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
    Assert(!res.V, "Malloc failed during Matrix Allocation.");

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
    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t j = 0;
                j < a.cols;
                j++)
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
    return (Matrix) {
        .rows = 1,
        .cols = a.cols,
        .V = &MatrixAT(a, row, 0),
    };
}

void
MatrixRandomize(RandomSeries* series, Matrix a, float low, float high)
{
    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t j = 0;
                j < a.cols;
                j++)
        {
            MatrixAT(a, i, j) = RandomBetween(series, low, high);
        }
    }
}

void
MatrixCopy(Matrix b, Matrix a)
{
    Assert(a.rows == b.rows, "");
    Assert(a.cols == b.cols, "");
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
    Assert(a.rows == b.rows, "");
    Assert(a.cols == b.cols, "");

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
    Assert(b.rows == a.rows, "");
    Assert(b.cols == a.cols, "");
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
    Assert(a.cols == b.rows, "");
    Assert(a.rows == c.rows, "");
    Assert(c.cols == b.cols, "");

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
        printf("\n");
    }
}

void
MatrixSigmoid(Matrix a)
{
    /*for (size_t i = 0; i < a.rows; i++) {*/
    /*    for (size_t j = 0; j < a.cols; j++) {*/
    /*        MatrixAT(a, i, j) = sigmoidf(MatrixAT(a, i, j));*/
    /*    }*/
    /*}*/
    MatrixApply(a, sigmoidf);
}

void
MatrixReLU(Matrix a)
{
    MatrixApply(a, reluf);
}

#endif
