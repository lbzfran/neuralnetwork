#ifndef MATRIX_H
#define MATRIX_H

// NOTE(liam): replace stdint include with base
#include <stdio.h>
#include <stdlib.h>

#include "alloc.h"
#include "random.h"
/*#define ArrayAccess(a, i, j) ((a)[(i) * ncolumns + (j)])*/

#ifndef m_alloc
#define m_alloc malloc
#define m_realloc realloc
#define m_free free
#endif

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

/*void MatrixCheckSameDim(Matrix, Matrix);*/
void MatrixAddScalar(Matrix, Matrix, float);
void MatrixAddMatrix(Matrix, Matrix, Matrix);
void MatrixDot(Matrix, Matrix, Matrix);
/*void MatrixTranspose(Matrix a, Matrix* b);*/
/*#define MatrixGet(A, i, j) ((A)[(i * A->cols + j)])*/

#endif

#define MATRIX_IMPLEMENTATION // debug purposes
#ifdef  MATRIX_IMPLEMENTATION

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
    res.V = (float*)malloc(sizeof(float) * rows * cols);
    Assert(!res.V, "Malloc failed during Matrix Allocation.");

    return res;
}

inline void
MatrixFree(Matrix a)
{
    if (a.V)
    {
        free(a.V);
    }
}

inline void
MatrixPrint(Matrix a)
{
    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t j = 0;
                j < a.cols;
                j++)
        {
            printf("%f ", MatrixAT(a, i, j));
        }
        printf("\n");
    }
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

inline int
MatrixCheckSameDim(Matrix* a, Matrix* b)
{
    if (a->cols == b->rows) return(0);
    return(1);
}

inline void
MatrixAddScalar(Matrix b, Matrix a, float x)
{
    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t col_idx = 0;
                col_idx < a.cols;
                col_idx++)
        {
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

    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t col_idx = 0;
                col_idx < b.cols;
                col_idx++)
        {
            /*c->V[i][col_idx] = a->V[i][col_idx] + b->V[i][col_idx];*/
            /*c.V[i * c.cols + col_idx] = a.V[i * a.cols + col_idx] + b.V[i * b.cols + col_idx];*/
            MatrixAT(c, i, col_idx) = MatrixAT(a, i, col_idx) + MatrixAT(b, i, col_idx);
        }
    }
}

inline void
MatrixSub(Matrix c, Matrix a, Matrix b)
{
    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t j = 0;
                j < b.cols;
                j++)
        {
            /*c->V[i][j] = a->V[i][j] - b->V[i][j];*/
            MatrixAT(c, i, j) = MatrixAT(a, i, j) - MatrixAT(b, i, j);
        }
    }
}

inline void
MatrixDot(Matrix c, Matrix a, Matrix b)
{
    // NOTE(liam): implied that both matrix dimensions are the same.
    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t j = 0;
                j < b.cols;
                j++)
        {
            float sum = 0;
            for (size_t k = 0;
                    k < b.rows;
                    k++)
            {
                /*sum += a.V[i * a.cols + k] * b.V[k * b.cols + j];*/
                sum += MatrixAT(a, i, k) * MatrixAT(b, i, j);
            }
            /*c->V[i * c->cols + j] = sum;*/
            MatrixAT(c, i, j) = sum;
        }
    }
}

inline Matrix
MatrixApply(Matrix a, float (*fun)(float))
{
    for (size_t i = 0;
            i < a.rows;
            i++)
    {
        for (size_t j = 0;
                j < a.cols;
                j++)
        {
            a.V[i * a.cols + j] = (*fun)(a.V[i* a.cols + j]);
        }
    }
    return(a);
}

inline void
MatrixTranspose(Matrix a, Matrix* b)
{
        for (size_t i = 0;
                i < b->rows;
                i++)
        {
            for (size_t j = 0;
                    j < b->cols;
                    j++)
            {
                b->V[i * b->cols + j] = a.V[j * a.cols + i];
            }
            printf("\n");
        }
}

#endif
