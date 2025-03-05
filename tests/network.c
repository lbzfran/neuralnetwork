
#define MATRIX_IMPLEMENTATION
#include "matrix.h"
#include <math.h>
#include "random.h"

#define ARENA_IMPLEMENTATION
#include "arena.h"

typedef struct NeuralNet {
    size_t layerCount;
    size_t *layerSizes;

    Matrix *W;
    Row *B;
} NeuralNet;

void NeuralNetInit(Arena* arena, RandomSeries *series, NeuralNet* nn, size_t *layerSizes, size_t layerCount)
{
    nn->layerCount = layerCount; // total # of layers
    nn->layerSizes = layerSizes; // # of neurons per layer
                                 // by standard, the first size determines
                                 // the input size, and the last determines
                                 // the output size.
                                 // This does mean that W and B will have
                                 // their 0th index point to the second size

    Matrix *W = PushArray(arena, Matrix, layerCount - 1);
    Row *B = PushArray(arena, Row, layerCount - 1);

    nn->W = W;
    nn->B = B;

    for (size_t l = 0; l < layerCount - 1; l++)
    {
        nn->W[l] = MatrixArenaAlloc(arena, layerSizes[l], layerSizes[l+1]);
        nn->B[l] = RowArenaAlloc(arena, layerSizes[l + 1]);

        // TODO(liam): should this be done by default?
        MatrixRandomize(series, nn->W[l], -1.f, 1.f);
        MatrixRandomize(series, nn->B[l], -1.f, 1.f);
    }
}

void NeuralNetForward(Arena *arena, NeuralNet nn, Row x)
{
    // NOTE(liam): a[l] = w * a[l-1] + b; a[0] = x
    ArenaTemp tmp = ArenaScratchCreate(arena);

    Matrix Z[nn.layerCount];
    Matrix A[nn.layerCount];
    for (uint32 l = 0; l < nn.layerCount - 1; l++)
    {
        Z[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);
        A[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);
    }

    MatrixDot(Z[0], x, nn.W[0]);

    printf("failing sizes: Z.col = %lu, W.col = %lu\n", Z->cols, nn.W->cols);
    /*MatrixAddMatrix(Z[0], Z[0], nn.B[0]);*/
    /*MatrixCopy(A[0], Z[0]);*/
    /*MatrixSigmoid(A[0]);*/

    for (uint32 l = 1; l < nn.layerCount - 1; l++)
    {
        /*MatrixDot(Z[l], A[l-1], nn.W[l]);*/
        /*MatrixAddMatrix(Z[l], Z[l], nn.B[l]);*/
        /*MatrixCopy(A[l], Z[l]);*/
        /*MatrixSigmoid(A[l]);*/
    }

    for (uint32 l = 0; l < nn.layerCount; l++)
    {
        MatrixPrint(A[l]);
    }

    ArenaScratchFree(tmp);
}

float32 NeuralNetCost(NeuralNet, Matrix);

void NeuralNetBackprop(NeuralNet);
void NeuralNetLearn(NeuralNet);

int main(void)
{
    Arena arena = {0};
    RandomSeries series = {0};
    RandomSeed(&series, 64);

    size_t layerCount = 12;
    size_t sizes[3] = {3, 8, 4};
    NeuralNet nn = {0};
    NeuralNetInit(&arena, &series, &nn, sizes, 3);

    for (int i = 0; i < 2; i++)
    {
        MatrixPrint(nn.W[i]);
        MatrixPrint(nn.B[i]);
    }

    Row x_train = RowArenaAlloc(&arena, 3);
    RowAT(x_train, 0) = 1;
    RowAT(x_train, 1) = 2;
    RowAT(x_train, 2) = 3;
    MatrixPrint(x_train);

    NeuralNetForward(&arena, nn, x_train);

    /*Matrix Z0 = MatrixArenaAlloc(&arena, 1, 8);*/
    /*Matrix A0 = MatrixArenaAlloc(&arena, 1, 8);*/
    /**/
    /*MatrixDot(Z0, x_train, nn.W[0]); // 1x8 = 1x3 * 3x8*/
    /*MatrixAddMatrix(Z0, Z0, nn.B[0]); // 1x8 = 1x8 + 1x8*/
    /*MatrixCopy(A0, Z0);*/
    /*MatrixSigmoid(A0);*/
    /**/
    /*MatrixPrint(Z0);*/
    /*MatrixPrint(A0);*/
    /**/
    /*Matrix Z1 = MatrixArenaAlloc(&arena, 1, 4); // (1x8 * 8x4) + 1x4*/
    /*Matrix A1 = MatrixArenaAlloc(&arena, 1, 4);*/
    /**/
    /*MatrixDot(Z1, A0, nn.W[1]); // 1x8 = 1x3 * 3x8*/
    /*MatrixAddMatrix(Z1, Z1, nn.B[1]); // 1x8 = 1x8 + 1x8*/
    /*MatrixCopy(A1, Z1);*/
    /*MatrixSigmoid(A1);*/
    /**/
    /*MatrixPrint(Z1);*/
    /*MatrixPrint(A1);*/

    ArenaFree(&arena);
    return 0;
}
