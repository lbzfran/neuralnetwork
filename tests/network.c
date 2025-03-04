
#define MATRIX_IMPLEMENTATION
#include "matrix.h"
#include <math.h>
#include "random.h"

#define ARENA_IMPLEMENTATION
#include "arena.h"

typedef struct NeuralNet {
    size_t layerCount;

    Matrix *W;
    Row *B;
} NeuralNet;

void NeuralNetInit(Arena* arena, NeuralNet* nn, size_t layerCount)
{
    nn->layerCount = layerCount;
    /*nn->W = PushArray(arena, Matrix, layerCount);*/
    /*nn->B = PushArray(arena, Row, layerCount);*/

    /*for (size_t i = 0; i < layerCount; i++)*/
    /*{*/
    /*    nn->W[i] = MatrixArenaAlloc(arena, layerCount, layerCount);*/
    /*    nn->B[i] = RowArenaAlloc(arena, layerCount);*/
    /*}*/
}

void NeuralNetForward(NeuralNet nn)
{
    // NOTE(liam): a[l] = w * a[l-1] + b; a[0] = x
}

float32 NeuralNetCost(NeuralNet, Matrix);

void NeuralNetBackprop(NeuralNet);
void NeuralNetLearn(NeuralNet);

int main(void)
{
    Arena arena = {0};

    size_t layerCount = 12;
    NeuralNet nn = {0};
    nn.W = PushArray(&arena, Matrix, layerCount);
    nn.B = PushArray(&arena, Matrix, layerCount);

    for (size_t i = 0; i < layerCount; i++)
    {
        nn.W[i] = MatrixArenaAlloc(&arena, layerCount, layerCount);
        nn.B[i] = RowArenaAlloc(&arena, layerCount);

        MatrixPrint(nn.W[i]);
    }

    /*NeuralNetInit(&arena, &nn, 10);*/

    ArenaFree(&arena);
    return 0;
}
