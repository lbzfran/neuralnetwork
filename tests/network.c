
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

size_t NNIndexSafe(NeuralNet nn, size_t layerNum, uint32 index)
{
    // NOTE(liam): safely index between layer sizes.
    if (layerNum > nn.layerCount - 2) printf("DEBUG: Truncating %lu layer to %lu\n", layerNum, nn.layerCount - 2);
    size_t limit = nn.layerSizes[ClampDown(layerNum, nn.layerCount - 2)] - 1;
    if (index > limit) printf("DEBUG: Truncating %d index to %lu\n", index, limit);
    return ClampDown(limit, index);
}

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
        nn->W[l] = MatrixArenaAlloc(arena, layerSizes[l], layerSizes[l + 1]);
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
    MatrixDot_(Z[0], x, nn.W[0]);
    MatrixAddM_(Z[0], Z[0], nn.B[0]);
    MatrixCopy_(A[0], Z[0]);
    MatrixSigmoid(A[0]);

    /*printf("sizes of misint:\n");*/
    /*printf("Z[%d] = {%lu, %lu}\n", 0, Z[0].rows, Z[0].cols);*/
    /*printf("x = {%lu, %lu}\n", x.rows, x.cols);*/
    /*printf("nn.W[%d] = {%lu, %lu}\n", 0, nn.W[0].rows, nn.W[0].cols);*/

    for (uint32 l = 1; l < nn.layerCount - 1; l++)
    {
        /*if (l == nn.layerCount - 2)*/
        /*{*/
        /*    printf("sizes of misint:\n");*/
        /*    printf("Z[%d] = {%lu, %lu}\n", l, Z[l].rows, Z[l].cols);*/
        /*    printf("A[%d] = {%lu, %lu}\n", l-1, A[l-1].rows, A[l-1].cols);*/
        /*    printf("nn.W[%d] = {%lu, %lu}\n", l, nn.W[l].rows, nn.W[l].cols);*/
        /*}*/

        MatrixDot_(Z[l], A[l-1], nn.W[l]);
        MatrixAddM_(Z[l], Z[l], nn.B[l]);
        MatrixCopy_(A[l], Z[l]);
        MatrixSigmoid(A[l]);
    }
}

float32
dsigmoidf(float32 z)
{
    return z * (1 - z);
}

float32 NeuralNetCost(NeuralNet, Matrix);

// uses sgd
void NeuralNetBackprop(Arena *arena, NeuralNet nn, Row x, Row y)
{
    NeuralHelper nh = {0};
    NeuralHelperInit(arena, &nh, nn);

    Matrix *dW = PushArray(arena, Matrix, nn.layerCount - 1);
    Row *dB = PushArray(arena, Row, nn.layerCount - 1);

    for (size_t l = 0; l < nn.layerCount - 1; l++)
    {
        dW[l] = MatrixArenaAlloc(arena, nn.layerSizes[l], nn.layerSizes[l + 1]);
        dB[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);

        // TODO(liam): this could be unnecessary.
        MatrixFill(dW[l], 0.0f);
        MatrixFill(dB[l], 0.0f);
    }

    NeuralNetForward(&nh, nn, x); // populates nh with A and Z

    // TODO(liam): adjust size; should y be a single element instead??
    // How many x per backprop.. 1 example? or a matrix of examples?
    // ANSWER: per SGD, only 1 example, with 1 output example.
    // input size: matrix of size (n examples) x (m data)
    // output size: row of size 1 to n; 1 for binary classification, and more
    // for non-binary

    /*size_t lastLayerSize = nn.layerSizes[nn.layerCount - 2] - 1;*/
    size_t lastLayerSize = NNIndexSafe(nn, nn.layerCount - 2, 999); // finds last layer, last index
    /*Row delta = RowArenaAlloc(arena, 1);*/
    /*Row dZ = RowArenaAlloc(arena, 1);*/
    Row dZ = MatrixCopy(arena, nh.Z[lastLayerSize]);

    /*printf("last size: %lu\n", lastLayerSize);*/

    Row delta = MatrixSubM(arena, nh.A[lastLayerSize], y);
    MatrixApply(dZ, dsigmoidf);
    MatrixMulM_(delta, delta, dZ);

    RowAT(dB[lastLayerSize], 0) = RowAT(delta, 0);

    Matrix A_T = MatrixTranspose(arena, nh.A[lastLayerSize - 1]);

    // TODO(liam): improper sizes
    MatrixDot_(dW[lastLayerSize], delta, A_T);

    // NOTE(liam): prints
    MatrixPrint(nh.A[lastLayerSize - 1]);
    MatrixPrint(x);
    MatrixPrint(y);
    MatrixPrint(dZ);
    MatrixPrint(delta);
    MatrixPrint(dW[lastLayerSize]);
    MatrixPrint(dB[lastLayerSize]);
    MatrixPrint(nh.A[lastLayerSize - 1]);
    MatrixPrint(A_T);
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
        MatrixShuffleCol(y_train, swapIdx, shuffleCount);

        // TODO(liam): continue here for learning algo
        for (int j = 0; j < n; j += batch_size)
        {
        }
    }

    /*MatrixPrint(x_train);*/
    /*MatrixPrint(y_train);*/
}

int main(void)
{
    Arena arena = {0};
    RandomSeries series = {0};
    RandomSeed(&series, 24323234);

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

    Matrix y_train = MatrixArenaAlloc(&arena, 4, 1);
    MatrixAT(y_train, 0, 0) = 0;
    MatrixAT(y_train, 1, 0) = 1;
    MatrixAT(y_train, 2, 0) = 1;
    MatrixAT(y_train, 3, 0) = 1;

    MatrixPrint(x_train);
    MatrixPrint(y_train);

    NeuralNetBackprop(&arena, nn, MatrixRow(x_train, 1), MatrixRow(y_train, 1));

    /*NeuralNetLearn(&series, nn, x_train, y_train, 1, 1);*/

    ArenaFree(&arena);
    return 0;
}
