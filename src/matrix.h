#ifndef MATRIX_H
#define MATRIX_H

// NOTE(liam): replace stdint include with base
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "alloc.h"
#include "random.h"

#ifndef m_alloc
#define m_alloc malloc
#define m_realloc realloc
#define m_free free
#endif

float sigmoidf(float x);

typedef struct Matrix {
    size_t rows;
    size_t cols;
    float* V;
} Matrix;

#define MatrixAT(m, i, j) ((m).V[(i) * m.cols + (j)])

Matrix MatrixAlloc(Arena* arena, size_t rows, size_t cols);
Matrix MatrixMalloc(size_t rows, size_t cols);
void MatrixFree(Matrix a);
void MatrixPrint(Matrix a);
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

#define MATRIX_IMPLEMENTATION // debug purposes
#ifdef  MATRIX_IMPLEMENTATION

inline float
sigmoidf(float x)
{
    // NOTE(liam): return value between 0 and 1 based on how far along
    // the given float is from -inf to inf.
    return 1.f / (1.f + expf(-x));
}

inline Matrix
MatrixAlloc(Arena* arena, size_t rows, size_t cols)
{
    Matrix res;

    res.rows = rows;
    res.cols = cols;
    res.V = PushArray(arena, float, rows * cols);

    return res;
}

inline Matrix
MatrixMalloc(size_t rows, size_t cols)
{
    Matrix res;

    res.rows = rows;
    res.cols = cols;
    res.V = (float*)m_alloc(sizeof(float) * rows * cols);
    Assert(!res.V, "Malloc failed during Matrix Allocation.");

    return res;
}

inline void
MatrixFree(Matrix a)
{
    if (a.V)
    {
        m_free(a.V);
    }
}

inline void
MatrixPrint(Matrix a)
{
    printf("[\n");
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

inline Matrix
MatrixRow(Matrix a, size_t row)
{
    return (Matrix) {
        .rows = 1,
        .cols = a.cols,
        .V = &MatrixAT(a, row, 0),
    };
}

inline void
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

inline void
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

inline void
MatrixFill(Matrix a, float x)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            MatrixAT(a, i, j) = x;
        }
    }
}

inline void
MatrixAddScalar(Matrix b, Matrix a, float x)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t col_idx = 0; col_idx < a.cols; col_idx++) {
            /*b->V[i][col_idx] = a->V[i][col_idx] + x;*/
            MatrixAT(b, i, col_idx) = MatrixAT(a, i, col_idx) + x;
        }
    }
}

inline void
MatrixAddMatrix(Matrix c, Matrix a, Matrix b)
{
    Assert(a.rows == b.rows, "");
    Assert(a.cols == b.cols, "");

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t col_idx = 0; col_idx < b.cols; col_idx++) {
            /*c->V[i][col_idx] = a->V[i][col_idx] + b->V[i][col_idx];*/
            /*c.V[i * c.cols + col_idx] = a.V[i * a.cols + col_idx] + b.V[i * b.cols
             * + col_idx];*/
            MatrixAT(c, i, col_idx) =
                MatrixAT(a, i, col_idx) + MatrixAT(b, i, col_idx);
        }
    }
}

inline void
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

inline void
MatrixSubScalar(Matrix b, Matrix a, float x)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t col_idx = 0; col_idx < a.cols; col_idx++) {
            /*b->V[i][col_idx] = a->V[i][col_idx] + x;*/
            MatrixAT(b, i, col_idx) = MatrixAT(a, i, col_idx) - x;
        }
    }
}

inline void
MatrixSubMatrix(Matrix c, Matrix a, Matrix b)
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            /*c->V[i][j] = a->V[i][j] - b->V[i][j];*/
            MatrixAT(c, i, j) = MatrixAT(a, i, j) - MatrixAT(b, i, j);
        }
    }
}

inline void
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

inline void
MatrixApply(Matrix a, float (*fun)(float))
{
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            a.V[i * a.cols + j] = (*fun)(a.V[i * a.cols + j]);
        }
    }
}

inline void
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

inline void
MatrixSigmoid(Matrix a)
{
    /*for (size_t i = 0; i < a.rows; i++) {*/
    /*    for (size_t j = 0; j < a.cols; j++) {*/
    /*        MatrixAT(a, i, j) = sigmoidf(MatrixAT(a, i, j));*/
    /*    }*/
    /*}*/
    MatrixApply(a, sigmoidf);
}

#endif
