#ifndef MATRIX_H
#define MATRIX_H

# define Swap(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
// NOTE(liam): replace stdint include with base
#include <stdio.h>
/*#define ArrayAccess(a, i, j) ((a)[(i) * ncolumns + (j)])*/

typedef struct matrix_t {
    size_t rows;
    size_t cols;
    int* V;
} Matrix;

float MatrixGet(Matrix*, int, int);
int MatrixCheckSameDim(Matrix* a, Matrix* b);
int MatrixAddScalar(Matrix* b, Matrix* a, float x);
int MatrixAddMatrix(Matrix* c, Matrix* a, Matrix* b);
int MatrixDot(Matrix* c, Matrix* a, Matrix* b);
void MatrixTranspose(Matrix* a);
/*#define MatrixGet(A, i, j) ((A)[(i * A->cols + j)])*/

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
            /*b->V[row_idx][col_idx] = a->V[row_idx][col_idx] + x;*/
            b->V[row_idx * b->cols + col_idx] = a->V[row_idx * a->cols + col_idx] + x;
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
            /*c->V[row_idx][col_idx] = a->V[row_idx][col_idx] + b->V[row_idx][col_idx];*/
            c->V[row_idx * c->cols + col_idx] = a->V[row_idx * a->cols + col_idx] + b->V[row_idx * b->cols + col_idx];
        }
    }
    return(0);
}

inline int
MatrixSub(Matrix* c, Matrix* a, Matrix* b)
{
    if (!MatrixCheckSameDim(a, b) || !MatrixCheckSameDim(b, c)) return(1);
    size_t size = a->cols;

    for (size_t row_idx = 0;
            row_idx < a->rows;
            row_idx++)
    {
        for (size_t col_idx = 0;
                col_idx < b->cols;
                col_idx++)
        {
            /*c->V[row_idx][col_idx] = a->V[row_idx][col_idx] - b->V[row_idx][col_idx];*/
            c->V[row_idx * size + col_idx] = a->V[row_idx * size + col_idx] - b->V[row_idx * size + col_idx];
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
                sum += a->V[a_row_idx * a->cols + b_row_idx] * b->V[b_row_idx * b->cols + b_col_idx];
            }
            c->V[a_row_idx * c->cols + b_col_idx] = sum;
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
            a.V[row_idx * a.cols + col_idx] = (*fun)(a.V[row_idx* a.cols + col_idx]);
        }
    }
    return(a);
}

inline void
MatrixTranspose(Matrix* a)
{
    for (size_t row_idx = 0;
            row_idx < a->rows;
            row_idx++)
    {
        for (size_t col_idx = 0;
                col_idx < a->cols;
                col_idx++)
        {
            // TODO(liam): perform swap.
            /*a->V[col_idx * a->rows + row_idx] = a->V[row_idx * a->cols + col_idx];*/
        }
    }
    Swap(a->rows, a->cols);
}

#endif
