#ifndef MATRIX_H
#define MATRIX_H

// NOTE(liam): replace stdint include with base
#include <stdio.h>
/*#define ArrayAccess(a, i, j) ((a)[(i) * ncolumns + (j)])*/

typedef struct matrix_t {
    size_t rows;
    size_t cols;
    float* V;
} Matrix;

inline int
MatrixCheckSameDim(Matrix* a, Matrix* b)
{
    if (a->cols == b->rows) return(0);
    return(1);
}

inline int
MatrixAddScalar(Matrix* b, Matrix* a, float x)
{
    if (!MatrixCheckSameDim(a, b)) return(1);

    for (size_t row_idx = 0;
            row_idx < a->rows;
            row_idx++)
    {
        for (size_t col_idx = 0;
                col_idx < a->cols;
                col_idx++)
        {
            b->V[row_idx][col_idx] = a->V[row_idx][col_idx] + x;
        }
    }
    return(0);
}
inline int
MatrixAddMatrix(Matrix* c, Matrix* a, Matrix* b)
{
    if (!MatrixCheckSameDim(a, b) || !MatrixCheckSameDim(b, c)) return(1);

    for (size_t row_idx = 0;
            row_idx < a->rows;
            row_idx++)
    {
        for (size_t col_idx = 0;
                col_idx < b->cols;
                col_idx++)
        {
            c->V[row_idx][col_idx] = a->V[row_idx][col_idx] + b->V[row_idx][col_idx];
        }
    }
    return(0);
}

inline int
MatrixSub(Matrix* c, Matrix* a, Matrix* b)
{
    if (!MatrixCheckSameDim(a, b) || !MatrixCheckSameDim(b, c)) return(1);

    for (size_t row_idx = 0;
            row_idx < a->rows;
            row_idx++)
    {
        for (size_t col_idx = 0;
                col_idx < b->cols;
                col_idx++)
        {
            c->V[row_idx][col_idx] = a->V[row_idx][col_idx] - b->V[row_idx][col_idx];
        }
    }
    return(0);
}

inline int
MatrixDot(Matrix* c, Matrix* a, Matrix* b)
{
    if (!MatrixCheckSameDim(a, b) || !MatrixCheckSameDim(b, c)) return(1);
    // NOTE(liam): implied that both matrix dimensions are the same.
    for (size_t a_row_idx = 0;
            a_row_idx < a->rows;
            a_row_idx++)
    {
        for (size_t b_col_idx = 0;
                b_col_idx < b->cols;
                b_col_idx++)
        {
            float sum = 0;
            for (size_t b_row_idx = 0;
                    b_row_idx < b->rows;
                    b_row_idx++)
            {
                sum += a->V[a_row_idx][b_row_idx] * b->V[b_row_idx][b_col_idx];
            }
            c->V[a_row_idx][b_col_idx] = sum;
        }
    }

    return(0);
}

inline Matrix
MatrixApply(Matrix a, float (*fun)(float))
{
    for (size_t row_idx = 0;
            row_idx < a.rows;
            row_idx++)
    {
        for (size_t col_idx = 0;
                col_idx < a.cols;
                col_idx++)
        {
            a.V[row_idx][col_idx] = (*fun)(a.V[row_idx][col_idx]);
        }
    }
    return(a);
}

inline Matrix
MatrixTranspose(Matrix a)
{
    for (size_t row_idx = 0;
            row_idx < a.rows;
            row_idx++)
    {
        for (size_t col_idx = 0;
                col_idx < a.cols;
                col_idx++)
        {
            a.V[col_idx][row_idx] = a.V[row_idx][col_idx];
        }
    }
    return(a);
}

#endif
