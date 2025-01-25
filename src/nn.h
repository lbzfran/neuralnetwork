/*
 * ---------------
 * Liam Bagabag
 * Version: 1.0.0
 * dependencies: matrix.h (direct)
 * ---------------
 */
#ifndef NRLNET_H
#define NRLNET_H

#include "alloc.h"
#include "base.h"
#include "matrix.h"

// shares the same allocator as matrix; i.e., m_alloc

typedef struct {
    int inputSize;
    int outputSize;

    Matrix w;
    Matrix b;

    Matrix output;
    Matrix dt;
} NrlLayer;

typedef struct {
    int capacity;
    int size;
    NrlLayer* layers;
} NrlNet;


inline void
NrlLayerAlloc(NrlLayer* layer,
        int inputSize, int outputSize,
        Matrix _alloced_w,
        Matrix _alloced_b,
        Matrix _alloced_outputs,
        Matrix _alloced_deltas);

NrlNet NrlNetAlloc(int, NrlLayer*);
#define NrlNetArenaAlloc(arena, n) NrlNetAlloc((n), PushArray((arena), NrlLayer, (n)))

NrlNet NrlNetMalloc(int);
void NrlNetFree(NrlNet);

// NOTE(liam): args: layer type, neuron amount, activation function
void NrlNetAddLayer(NrlNet, int, int, float(*)(float));
void NrlNetInit(NrlNet net, int* layerSizes);

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

#define NRLNET_IMPLEMENTATION
#ifdef NRLNET_IMPLEMENTATION

inline NrlNet
NrlNetAlloc(int capacity, NrlLayer* layers)
{
    NrlNet res = (NrlNet) {
        .capacity = capacity,
        .size = 0,
        .layers = layers
    };

    return res;
}

inline NrlNet
NrlNetMalloc(int capacity)
{
    NrlNet res = (NrlNet) {
        .size = 0,
        .layers = (NrlLayer*)m_alloc(sizeof(NrlLayer) * capacity)
    };

    return res;
}

inline void
NrlNetFree(NrlNet net)
{
    m_free(net.layers);
}

//NOTE(liam): allocates a single layer.
inline void
NrlNetLayerAlloc(RandomSeries* local_series,
        NrlLayer* layer,
        int inputSize, int outputSize,
        Matrix _alloced_w,
        Matrix _alloced_b,
        Matrix _alloced_outputs,
        Matrix _alloced_deltas)
{
    layer->inputSize = inputSize;
    layer->outputSize = outputSize;

    layer->w = _alloced_w;
    layer->b = _alloced_b;

    layer->output = _alloced_outputs;
    layer->dt = _alloced_deltas;

    /*MatrixRandomize(RandomSeries *series, Matrix a, float low, float high)*/
    MatrixRandomize(local_series, layer->w, 0, 1);
    MatrixRandomize(local_series, layer->b, 0, 1);
    MatrixFill(layer->output, 0.f);
    MatrixFill(layer->dt, 0.f);
}

// NOTE(liam); initializes layers for nrlnet.
inline void
NrlNetInit(RandomSeries* local_series, Arena* arena, NrlNet net, int* layerSizes)
{
    NrlLayer* layer = net.layers + net.size;
    for (int i = 0; i < net.capacity; i++) {

        NrlNetLayerAlloc(local_series,
            layer, layerSizes[i], layerSizes[i+1],
            MatrixArenaAlloc(arena, 2, 2),
            MatrixArenaAlloc(arena, 2, 2),
            MatrixArenaAlloc(arena, 2, 2),
            MatrixArenaAlloc(arena, 2, 2)
        );
    }
}

inline int
somemain(void)
{
    Arena* arena = ArenaMalloc(Kilobytes(1));

    NrlNet net = NrlNetArenaAlloc(arena, 2);

    int layerSizes[2] = {
        2, 2
    };

    NrlNetInit(net, layerSizes);

    return 0;
}

// TODO(liam): do this + free.
/*inline void*/
/*NrlNetLayerMalloc(NrlLayer* layer,*/
/*        int inputSize, int outputSize,*/
/*        Matrix _alloced_w,*/
/*        Matrix _alloced_b,*/
/*        Matrix _alloced_outputs,*/
/*        Matrix _alloced_deltas)*/
/*{*/
/*    layer->inputSize = inputSize;*/
/*    layer->outputSize = outputSize;*/
/**/
/*    layer->W = _alloced_w;*/
/*    layer->B = _alloced_b;*/
/**/
/*    layer->output = _alloced_outputs;*/
/*    layer->dt = _alloced_deltas;*/
/*}*/

/*inline void*/
/*NrlNetAddLayer(NrlNet net, int layerType, int n, float(*activation)(float))*/
/*{*/
/*    NrlLayer* layers = net.layers;*/
/**/
/*    if (layerType == 0) // Deeep*/
/*    {*/
/**/
/*    }*/
/*}*/

#endif
