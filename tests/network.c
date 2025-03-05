
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

// NOTE(liam): this will only exist inside functions pertaining to the
// NeuralNet struct, so the size will always be derived from there.
typedef struct NeuralHelper {
    Row *Z;
    Row *A;
} NeuralHelper;

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

void NeuralHelperInit(Arena *arena, NeuralHelper *nh, NeuralNet nn)
{
    nh->Z = PushArray(arena, Row, nn.layerCount - 1);
    nh->A = PushArray(arena, Row, nn.layerCount - 1);

    for (uint32 l = 0; l < nn.layerCount - 1; l++)
    {
        nh->Z[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);
        nh->A[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);
    }
}

void NeuralNetForward(NeuralHelper *nh, NeuralNet nn, Row x)
{
    // NOTE(liam): a[l] = w * a[l-1] + b; a[0] = x
    Row *Z = nh->Z;
    Row *A = nh->A;

    // NOTE(liam): a[1] = z; z = w * a[0] + b
    MatrixDot(Z[0], x, nn.W[0]);
    MatrixAddMatrix(Z[0], Z[0], nn.B[0]);
    MatrixCopy(A[0], Z[0]);
    MatrixSigmoid(A[0]);

    for (uint32 l = 1; l < nn.layerCount - 1; l++)
    {
        MatrixDot(Z[l], A[l-1], nn.W[l]);
        MatrixAddMatrix(Z[l], Z[l], nn.B[l]);
        MatrixCopy(A[l], Z[l]);
        MatrixSigmoid(A[l]);
    }
}

float32 NeuralNetCost(NeuralNet, Matrix);

void NeuralNetBackprop(Arena *arena, NeuralNet nn, Row x)
{
    NeuralHelper nh = {0};
    NeuralHelperInit(arena, &nh, nn);

    NeuralNetForward(&nh, nn, x); // populates nh with A and Z

}

void NeuralNetLearn(RandomSeries *series,
        NeuralNet nn, Matrix x_train, Row y_train,
        size_t epochs, size_t batch_size)
{
    size_t n = x_train.rows * x_train.cols;
    for (size_t i = 0; i < epochs; i++)
    {
        size_t shuffleCount = x_train.cols;
        size_t swapIdx[shuffleCount * 2];

        MatrixRandomShuffleRow(series, x_train, shuffleCount, swapIdx);

        /*printf("values of swapIdx\n");*/
        /*for (int j = 0; j < x_train.cols * 2; j++)*/
        /*{*/
        /*    printf("%d\n", swapIdx[j]);*/
        /*}*/

        MatrixShuffleCol(y_train, swapIdx, shuffleCount);

        // TODO(liam): continue here for learning algo
    }

    MatrixPrint(x_train);
    MatrixPrint(y_train);
}

int main(void)
{
    Arena arena = {0};
    RandomSeries series = {0};
    RandomSeed(&series, 123423213);

    size_t layerCount = 12;
    size_t sizes[] = {2, 8, 6, 4, 1};
    NeuralNet nn = {0};
    NeuralNetInit(&arena, &series, &nn, sizes, ArrayCount(sizes));

    /*for (int i = 0; i < ArrayCount(sizes) - 1; i++)*/
    /*{*/
        /*MatrixPrint(nn.W[i]);*/
        /*MatrixPrint(nn.B[i]);*/
    /*}*/

    // TODO(liam): super inefficient assignments, might rework later
    Matrix x_train = MatrixArenaAlloc(&arena, 4, 2);
    MatrixAT(x_train, 0, 0) = 0;
    MatrixAT(x_train, 0, 1) = 0;
    MatrixAT(x_train, 1, 0) = 0;
    MatrixAT(x_train, 1, 1) = 1;
    MatrixAT(x_train, 2, 0) = 1;
    MatrixAT(x_train, 2, 1) = 0;
    MatrixAT(x_train, 3, 0) = 1;
    MatrixAT(x_train, 3, 1) = 1;

    Row y_train = RowArenaAlloc(&arena, 4);
    RowAT(y_train, 0) = 0;
    RowAT(y_train, 1) = 1;
    RowAT(y_train, 2) = 1;
    RowAT(y_train, 3) = 1;
    /*RowAT(x_train, 2) = 3;*/
    MatrixPrint(x_train);
    MatrixPrint(y_train);

    /*NeuralNetBackprop(&arena, nn, x_train);*/

    NeuralNetLearn(&series, nn, x_train, y_train, 1, 1);

    ArenaFree(&arena);
    return 0;
}
