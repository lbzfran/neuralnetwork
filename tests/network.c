
#define MATRIX_IMPLEMENTATION
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
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
typedef struct NeuralForward {
    Row *Z;
    Row *A;
} NeuralForward;

typedef struct NeuralBack {
    Matrix *dW;
    Row *dB;
} NeuralBack;


float32
dsigmoidf(float32 z)
{
    return z * (1 - z);
}

float32
dreluf(float32 z)
{
    return z >= 0 ? 1 : 0;
}

float32
squaref(float32 x)
{
    return x * x;
}

void NeuralNetUpdate(Arena *arena, NeuralNet nn, Matrix x_train, Matrix y_train, size_t exampleCount, float32 rate);

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

void NeuralHelperInit(Arena *arena, NeuralForward *nh, NeuralNet nn)
{
    nh->Z = PushArray(arena, Row, nn.layerCount - 1);
    nh->A = PushArray(arena, Row, nn.layerCount - 1);

    for (uint32 l = 0; l < nn.layerCount - 1; l++)
    {
        nh->Z[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);
        nh->A[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);
    }
}

void NeuralNetForward(NeuralForward *nh, NeuralNet nn, Row x)
{
    // NOTE(liam): a[l] = w * a[l-1] + b; a[0] = x
    Row *Z = nh->Z;
    Row *A = nh->A;

    // NOTE(liam): a[1] = z; z = w * a[0] + b
    MatrixDot_(Z[0], x, nn.W[0]);
    MatrixAddM_(Z[0], Z[0], nn.B[0]);
    MatrixCopy_(A[0], Z[0]);
    MatrixApply(A[0], sigmoidf);

    /*MatrixPrint(Z[0]);*/
    /*MatrixPrint(A[0]);*/
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
        MatrixApply(A[l], sigmoidf);

        /*MatrixPrint(Z[l]);*/
        /*MatrixPrint(A[l]);*/
    }

}

float32 NeuralNetCost(NeuralNet, Matrix);

// uses sgd
NeuralBack NeuralNetBackprop(Arena *arena, NeuralNet nn, Row x, Row y)
{
    NeuralForward nh = {0};
    NeuralHelperInit(arena, &nh, nn);

    NeuralBack nb = {0};

    nb.dW = PushArray(arena, Matrix, nn.layerCount - 1);
    nb.dB = PushArray(arena, Row, nn.layerCount - 1);

    Matrix *dW = nb.dW;
    Matrix *dB = nb.dB;

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

    // NOTE(liam): delta = (A[-1] - y) * dsigmoidf(Z[-1])
    size_t pos = nn.layerCount - 2;

    Row dZ = MatrixCopy(arena, nh.A[pos]);
    MatrixApply(dZ, dsigmoidf);

    // NOTE(liam): cost function
    // TODO(liam): likely fix the cost function application
    Row error = MatrixSubM(arena, nh.A[pos], y);

    /*MatrixPrint(x);*/
    /*MatrixPrint(nh.A[pos]);*/
    /*MatrixPrint(y);*/

    MatrixMulS_(error, error, 2.0f);
    /*MatrixPrint_(error, "error_squared");*/
    /*MatrixApply(error, squaref);*/

    Row delta = MatrixMulM(arena, error, dZ);

    if (isnan(RowAT(delta, 0)))
    {
        fprintf(stderr, "ERROR: NaN value detected in delta.\n");
        exit(1);
    }
    else if (RowAT(delta, 0) == 0.f)
    {
        fprintf(stderr, "zero training achieved..?\n");
    }
    MatrixPrint_(delta, "cost");

    MatrixCopy_(dB[pos], delta);

    // NOTE(liam): delta * A[-2].transpose()
    MatrixDot_(dW[pos], MatrixTranspose(arena, nh.A[pos - 1]), delta);

    // LAYERS: { 2, 18, 1 }
    //           ^
    // WEIGHT SIZES: { 2x18, 18x1 }
    //                  ^
    // OUTPUT SIZES: { 1x18, 1x1 }
    {
        /*printf("sizes of misint:\n");*/
        /*printf("dW[%lu] = {%lu, %lu}\n", pos, dW[pos].rows, dW[pos].cols);*/
        /**/
        /*printf("nh.A_T[%lu] = {%lu, %lu}\n", pos, nh.A[pos].cols, nh.A[pos].rows);*/
        /*printf("delta = {%lu, %lu}\n", delta.rows, delta.cols);*/
    }

    /*for (uint32 l = nn.layerCount - 3; l > 0; l--)*/
    while (pos--)
    {
        dZ = MatrixCopy(arena, nh.A[pos]);
        MatrixApply(dZ, dsigmoidf);

        error = MatrixDot(arena, delta, MatrixTranspose(arena, nn.W[pos + 1]));
        /*MatrixApply(error, squaref);*/
        MatrixMulS_(error, error, 2.0f);
        delta = MatrixMulM(arena, error, dZ);

        /*RowAT(dB[l], 0) = RowAT(delta, 0);*/
        MatrixCopy_(dB[pos], delta);

        if (pos > 0)
        {
            MatrixDot_(dW[pos], MatrixTranspose(arena, nh.A[pos - 1]), delta);
        }
        else
        {
            // use x on the last iteration at first layer
            MatrixDot_(dW[pos], MatrixTranspose(arena, x), delta);
        }
        /*{*/
        /*    printf("sizes of misint:\n");*/
        /*    printf("dW[%d] = {%lu, %lu}\n", pos, dW[pos].rows, dW[pos].cols);*/
        /**/
        /*    printf("nh.A_T[%d] = {%lu, %lu}\n", pos, nh.A[pos].cols, nh.A[pos].rows);*/
        /*    printf("delta = {%lu, %lu}\n", delta.rows, delta.cols);*/
        /*    printf("x = {%lu, %lu}\n", x.rows, x.cols);*/
        /**/
        /*}*/
    }

    return nb;
}


void NeuralNetLearn(Arena *arena, RandomSeries *series,
        NeuralNet nn, Matrix x_train, Matrix y_train,
        size_t epochs, float32 rate, size_t batch_size)
{
    ArenaTemp tmp = ArenaScratchCreate(arena);

    size_t n = x_train.rows * x_train.cols;
    size_t actualBatchCount = (int)(n / batch_size);
    Matrix x_batches[actualBatchCount];
    Matrix y_batches[actualBatchCount];

    for (size_t e = 0; e < epochs; e++)
    {
        /*size_t shuffleCount = x_train.cols - 1;*/
        /*size_t swapIdx[shuffleCount * 2];*/
        /*MatrixRandomShuffleRow(series, x_train, shuffleCount, swapIdx);*/
        /*MatrixShuffleCol(y_train, swapIdx, y_train.cols - 1);*/

        /*if (batch_size == 1)*/
        /*{*/
        /*    NeuralNetUpdate(arena, nn, x_train, y_train, batch_size, rate);*/
        /*}*/
        /*else*/
        {
            for (int j = 0; j < n; j += batch_size)
            {
                x_batches[j] = MatrixSliceRow(arena, x_train, j, j + batch_size);
                y_batches[j] = MatrixSliceRow(arena, y_train, j, j + batch_size);
            }

            for (int j = 0; j < actualBatchCount; j++)
            {
                NeuralNetUpdate(arena, nn, x_batches[j], y_batches[j], batch_size, rate);
            }
        }


        /*printf("Epoch %lu completed.\n", e);*/
    }

    ArenaScratchFree(tmp);
}



void NeuralNetUpdate(Arena *arena, NeuralNet nn,
                     Matrix x_train, Matrix y_train,
                     size_t exampleCount, float32 rate)
{
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

    for (uint32 i = 0; i < exampleCount; i++)
    {
        NeuralBack nb = NeuralNetBackprop(arena, nn,
                                          MatrixRow(x_train, i),
                                          MatrixRow(y_train, i));

        for (uint32 j = 0; j < nn.layerCount - 1; j++)
        {
            /*MatrixPrint(nb.dB[j]);*/
            /*MatrixPrint(nb.dW[j]);*/
            MatrixSum(dB[j], nb.dB[j]);
            MatrixSum(dW[j], nb.dW[j]);
        }
    }

    for (uint32 i = 0; i < nn.layerCount - 1; i++)
    {
        /*MatrixSubS_(nn.W[i], nn.W[i], (rate / exampleCount));*/
        /*MatrixSubM_(nn.W[i], nn.W[i], dW[i]);*/
        float32 actualRate = rate / exampleCount;

        MatrixMulS_(dW[i], dW[i], actualRate);
        MatrixSubM_(nn.W[i], nn.W[i], dW[i]);

        MatrixMulS_(dB[i], dB[i], actualRate);
        MatrixSubM_(nn.B[i], nn.B[i], dB[i]);
    }
}

int main(void)
{
    Arena arena = {0};
    RandomSeries series = {0};
    RandomSeed(&series, 232134);

    size_t sizes[] = {2, 8, 16, 8, 1};
    NeuralNet nn = {0};
    NeuralNetInit(&arena, &series, &nn, sizes, ArrayCount(sizes));

    /*for (int i = 0; i < ArrayCount(sizes) - 1; i++)*/
    {
        /*MatrixPrint(nn.W[i]);*/
        /*MatrixPrint(nn.B[i]);*/
    }

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
    MatrixAT(y_train, 1, 0) = 0;
    MatrixAT(y_train, 2, 0) = 0;
    MatrixAT(y_train, 3, 0) = 1;

    /*MatrixPrint(x_train);*/
    /*MatrixPrint(y_train);*/

    /*NeuralNetBackprop(&arena, nn, MatrixRow(x_train, 1), MatrixRow(y_train, 1));*/


    /*NeuralNetUpdate(&arena, nn, x_train, y_train, 4, 0.001);*/

    NeuralNetLearn(&arena, &series, nn, x_train, y_train, 10000, 0.001, 1);

    NeuralForward nh = {0};
    NeuralHelperInit(&arena, &nh, nn);

    MatrixPrint(x_train);

    NeuralNetForward(&nh, nn, MatrixRow(x_train, 0));
    MatrixPrint(MatrixRow(x_train, 0));
    MatrixPrint(nh.A[nn.layerCount - 2]);
    NeuralNetForward(&nh, nn, MatrixRow(x_train, 1));
    MatrixPrint(MatrixRow(x_train, 1));
    MatrixPrint(nh.A[nn.layerCount - 2]);
    NeuralNetForward(&nh, nn, MatrixRow(x_train, 2));
    MatrixPrint(MatrixRow(x_train, 2));
    MatrixPrint(nh.A[nn.layerCount - 2]);
    NeuralNetForward(&nh, nn, MatrixRow(x_train, 3));
    MatrixPrint(MatrixRow(x_train, 3));
    MatrixPrint(nh.A[nn.layerCount - 2]);

    /*for (int i = 0; i < nn.layerCount - 1; i++)*/
    /*{*/
    /*    MatrixPrint(nh.A[i]);*/
    /*}*/
    /*for (int i = 0; i < ArrayCount(sizes) - 1; i++)*/
    {
        /*MatrixPrint(nn.W[i]);*/
        /*MatrixPrint(nn.B[i]);*/
    }

    ArenaFree(&arena);
    return 0;
}
