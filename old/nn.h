/*
 * ---------------
 * Liam Bagabag
 * Version: 1.0.0
 * dependencies: matrix.h (direct), random.h (direct)
 * requires: nn.c
 * ---------------
 */
#ifndef NRLNET_H
#define NRLNET_H

#include "arena.h"
#include "base.h"
#include "matrix.h"

// shares the same allocator as matrix; i.e., m_alloc

typedef struct NrlLayer {
    int inputSize;
    int outputSize;

    Matrix w;
    Matrix b;

    Matrix output;
    Matrix dt;
} NrlLayer;

typedef struct NrlNet {
    int capacity;
    int size;
    NrlLayer* layers;
} NrlNet;

NrlNet NrlNetAlloc(int, NrlLayer*);
#define NrlNetArenaAlloc(arena, n) (NrlNetAlloc((n), PushArray((arena), NrlLayer, (n))))
#define NrlNetMalloc(n) NrlNetAlloc((n), (NrlLayer*)m_alloc(sizeof(NrlLayer) * (n)))
/*NrlNet NrlNetMalloc(int);*/
void NrlNetFree(NrlNet);

void
NrlLayerAlloc(RandomSeries* local_series,
        NrlLayer* layer,
        int inputSize, int outputSize,
        Matrix _alloced_w,
        Matrix _alloced_b,
        Matrix _alloced_outputs,
        Matrix _alloced_deltas);

// NOTE(liam): args: layer type, neuron amount, activation function
void NrlNetAddLayer(NrlNet, int, int, float(*)(float));
void NrlNetInit(RandomSeries* local_series, Arena* arena, NrlNet net, int* layerSizes);

void NrlNetPrint(NrlNet);

// dot, activation function
void forward_();
void NrlNetForward(NrlNet);

#define EPOCH
#define RATE
float lossMSE(float, float);
float lossCrossEntropy(float, float);

void loss_();
void NrlNetBackPropagate(NrlNet);

#endif
