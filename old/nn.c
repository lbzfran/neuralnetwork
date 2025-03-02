#define RAND_IMPLEMENTATION
#define MATRIX_IMPLEMENTATION
#include "nn.h"

NrlNet
NrlNetAlloc(int capacity, NrlLayer* layers)
{
    NrlNet res = (NrlNet) {
        .capacity = capacity,
        .size = 0,
        .layers = layers
    };

    return res;
}

void
NrlNetFree(NrlNet net)
{
    m_free(net.layers);
}

//NOTE(liam): allocates a single layer.
void
NrlLayerAlloc(RandomSeries* local_series,
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
void
NrlNetInit(RandomSeries* local_series, Arena* arena, NrlNet net, int* layerSizes)
{
    NrlLayer* layer = net.layers + net.size;
    for (int i = 0; i < net.capacity; i++) {

        NrlLayerAlloc(local_series,
            layer, layerSizes[i], layerSizes[i+1],
            MatrixArenaAlloc(arena, 2, 2),
            MatrixArenaAlloc(arena, 2, 2),
            MatrixArenaAlloc(arena, 2, 2),
            MatrixArenaAlloc(arena, 2, 2)
        );
    }
}

void
NrlNetPrint(NrlNet net)
{
    for (size_t i = 0; i < net.size; i++) {
        NrlLayer* layer = net.layers + i;

        MatrixPrint(layer->w);
        MatrixPrint(layer->b);
        MatrixPrint(layer->output);
        MatrixPrint(layer->dt);
    }
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
