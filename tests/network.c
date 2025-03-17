
#define MATRIX_IMPLEMENTATION
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include "random.h"

#ifdef WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

#define ARENA_IMPLEMENTATION
#include "arena.h"

typedef struct NeuralNet {
    uint32 layerCount;
    uint32 *layerSizes;

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
dsigmoidf(float32 x)
{
    return sigmoidf(x) * (1 - sigmoidf(x));
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

void NeuralNetUpdate(Arena *arena, NeuralNet nn, Matrix x_train, Matrix y_train, uint32 exampleCount, float32 rate);

uint32 NNIndexSafe(NeuralNet nn, uint32 layerNum, uint32 index)
{
    // NOTE(liam): safely index between layer sizes.
    if (layerNum > nn.layerCount - 2) printf("DEBUG: Truncating %lu layer to %lu\n", layerNum, nn.layerCount - 2);
    uint32 limit = nn.layerSizes[ClampDown(layerNum, nn.layerCount - 2)] - 1;
    if (index > limit) printf("DEBUG: Truncating %d index to %lu\n", index, limit);
    return ClampDown(limit, index);
}

bool32 NeuralNetSave(NeuralNet nn, char *path)
{
    // saves the neural network to a csv-like file
    // to be loaded later.
    bool32 result = true;


    FILE *fp = fopen(path, "wb+");
    if (fp == NULL)
    {
        result = false;
    }
    else
    {
        size_t bytesWritten = 0;
        bytesWritten += fwrite(&nn.layerCount, sizeof(uint32), 1, fp);
        bytesWritten += fwrite(nn.layerSizes, sizeof(uint32), nn.layerCount, fp);

        // TODO(liam): writing matrices.
        for (uint32 l = 0; l < nn.layerCount - 1; l++)
        {
            /*dW[l] = MatrixArenaAlloc(arena, nn.layerSizes[l], nn.layerSizes[l + 1]);*/
            /*dB[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);*/
            bytesWritten += fwrite(nn.W[l].V, sizeof(float32), nn.layerSizes[l] * nn.layerSizes[l + 1], fp);
            bytesWritten += fwrite(nn.B[l].V, sizeof(float32), nn.layerSizes[l + 1], fp);
        }

        fclose(fp);
    }

    return result;
}

bool32 NeuralNetLoad(Arena *arena, NeuralNet *nn, char *path)
{
    bool32 result = true;

    FILE *fp = fopen(path, "rb");
    if (fp == NULL)
    {
        result = false;
    }
    else
    {
        fread(&nn->layerCount, sizeof(uint32), 1, fp);

        nn->layerSizes = PushArray(arena, uint32, nn->layerCount);
        fread(nn->layerSizes, sizeof(uint32), nn->layerCount, fp);

        Matrix *W = PushArray(arena, Matrix, nn->layerCount - 1);
        Row *B = PushArray(arena, Row, nn->layerCount - 1);

        nn->W = W;
        nn->B = B;

        for (uint32 l = 0; l < nn->layerCount - 1; l++)
        {
            nn->W[l] = MatrixArenaAlloc(arena, nn->layerSizes[l], nn->layerSizes[l + 1]);
            nn->B[l] = RowArenaAlloc(arena, nn->layerSizes[l + 1]);

            fread(W[l].V, sizeof(float32), nn->layerSizes[l] * nn->layerSizes[l + 1], fp);
            fread(B[l].V, sizeof(float32), nn->layerSizes[l + 1], fp);
        }

        fclose(fp);
    }

    return result;
}

void NeuralNetInit(Arena* arena, RandomSeries *series, NeuralNet *nn, uint32 *layerSizes, uint32 layerCount, bool32 randomize_params)
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

    for (uint32 l = 0; l < layerCount - 1; l++)
    {
        nn->W[l] = MatrixArenaAlloc(arena, layerSizes[l], layerSizes[l + 1]);
        nn->B[l] = RowArenaAlloc(arena, layerSizes[l + 1]);

        if (randomize_params)
        {
            MatrixRandomize(series, nn->W[l], -1.f, 1.f);
            MatrixRandomize(series, nn->B[l], -1.f, 1.f);
        }
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

    for (uint32 l = 1; l < nn.layerCount - 1; l++)
    {
        MatrixDot_(Z[l], A[l-1], nn.W[l]);
        MatrixAddM_(Z[l], Z[l], nn.B[l]);
        MatrixCopy_(A[l], Z[l]);
        MatrixApply(A[l], sigmoidf);
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

    for (uint32 l = 0; l < nn.layerCount - 1; l++)
    {
        dW[l] = MatrixArenaAlloc(arena, nn.layerSizes[l], nn.layerSizes[l + 1]);
        dB[l] = RowArenaAlloc(arena, nn.layerSizes[l + 1]);

        MatrixFill(dW[l], 0.0f);
        MatrixFill(dB[l], 0.0f);
    }

    NeuralNetForward(&nh, nn, x); // populates nh with A and Z

    // per SGD, only 1 example, with 1 output example.
    // input size: matrix of size (n examples) x (m data)
    // output size: row of size 1 to n; 1 for binary classification, and more
    // for non-binary

    // NOTE(liam): delta = (A[-1] - y) * dsigmoidf(Z[-1])
    uint32 pos = nn.layerCount - 2;

    Row dZ = MatrixCopy(arena, nh.Z[pos]);
    MatrixApply(dZ, dsigmoidf);

    // NOTE(liam): cost function
    // TODO(liam): likely fix the cost function application
    Row error = MatrixSubM(arena, nh.A[pos], y);
    MatrixMulS_(error, error, 2.0f);

    Row delta = MatrixMulM(arena, error, dZ);

    MatrixPrint_(delta, "cost");

    bool32 can_descend = true;
    if (isnan(RowAT(delta, 0)))
    {
        fprintf(stderr, "ERROR: NaN value detected in loss.\n");
        can_descend = false;
    }
    else if (RowAT(delta, 0) == 0.f)
    {
        fprintf(stderr, "WARNING: Network is not learning (aka. achieving 0 loss).\n");
        can_descend = false;
    }

    if (can_descend)
    {
        MatrixCopy_(dB[pos], delta);

        // NOTE(liam): delta * A[-2].transpose()
        MatrixDot_(dW[pos], MatrixTranspose(arena, nh.A[pos - 1]), delta);

        MatrixPrint(dW[pos]);

        // LAYERS: { 2, 18, 1 }
        //           ^
        // WEIGHT SIZES: { 2x18, 18x1 }
        //                  ^
        // OUTPUT SIZES: { 1x18, 1x1 }

        while (pos--)
        {
            dZ = MatrixCopy(arena, nh.A[pos]);
            MatrixApply(dZ, dsigmoidf);

            error = MatrixDot(arena, delta, MatrixTranspose(arena, nn.W[pos + 1]));
            MatrixMulS_(error, error, 2.0f);
            delta = MatrixMulM(arena, error, dZ);

            MatrixCopy_(dB[pos], delta);

            if (pos)
            {
                MatrixDot_(dW[pos], MatrixTranspose(arena, nh.A[pos - 1]), delta);
            }
            else
            {
                // use x on the last iteration at first layer
                MatrixDot_(dW[pos], MatrixTranspose(arena, x), delta);
            }
        }
    }

    return nb;
}


void NeuralNetLearn(Arena *arena, RandomSeries *series,
        NeuralNet nn, Matrix x_train, Matrix y_train,
        uint32 epochs, float32 rate, uint32 batch_size)
{
    ArenaTemp tmp = ArenaScratchCreate(arena);

    uint32 n = x_train.rows * x_train.cols;
    uint32 actualBatchCount = (int)(n / batch_size);
    Matrix x_batches[actualBatchCount];
    Matrix y_batches[actualBatchCount];

    for (uint32 e = 0; e < epochs; e++)
    {
        /*uint32 shuffleCount = x_train.cols - 1;*/
        /*uint32 swapIdx[shuffleCount * 2];*/
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
                     uint32 exampleCount, float32 rate)
{
    Matrix *dW = PushArray(arena, Matrix, nn.layerCount - 1);
    Row *dB = PushArray(arena, Row, nn.layerCount - 1);

    for (uint32 l = 0; l < nn.layerCount - 1; l++)
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
    RandomSeed(&series, 232122);

    size_t sizes[] = {2, 8, 6, 8, 6, 1};
    NeuralNet nn = {0};
    NeuralNetInit(&arena, &series, &nn, sizes, ArrayCount(sizes));

    // TODO(liam): inefficient assignments, might rework later
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
    MatrixAT(y_train, 3, 0) = 0;

    uint32 sizes[] = {2, 16, 32, 24, 1};
    NeuralNet nn = {0};

    if (access("hehe.bin", F_OK) == 0)
    {
        NeuralNetLoad(&arena, &nn, "hehe.bin");

        printf("matrix layer count: %d\n", nn.layerCount);
        for (uint32 l = 0; l < nn.layerCount; l++)
        {
            printf("matrix layer %d - size: %d\n", l, nn.layerSizes[l]);
        }
    }
    else
    {
        NeuralNetInit(&arena, &series, &nn, sizes, ArrayCount(sizes), true);
        NeuralNetLearn(&arena, &series, nn, x_train, y_train, 10000, 0.01, 1);
        NeuralNetSave(nn, "hehe.bin");
    }

    NeuralForward nh = {0};
    NeuralHelperInit(&arena, &nh, nn);

    MatrixPrint(x_train);

    for (uint32 i = 0; i < y_train.rows; i++)
    {
        NeuralNetForward(&nh, nn, MatrixRow(x_train, i));
        MatrixPrint(MatrixRow(x_train, i));
        MatrixPrint(nh.A[nn.layerCount - 2]);
    }

    ArenaFree(&arena);
    return 0;
}
