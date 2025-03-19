#include "matrix.h"
#include <math.h>
#include "random.h"

#ifdef WIN32
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#endif

#include "arena.h"

typedef struct NeuralNet {
    uint32 layerCount;
    uint32 layerCapacity;
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

float32 sigmoidf(float32 x);
float32 dsigmoidf(float32 z);

float32 reluf(float32 x);
float32 dreluf(float32 z);

uint32 NeuralNetIndexSafe(NeuralNet nn, uint32 layerNum, uint32 index);
void NeuralHelperInit(Arena *arena, NeuralForward *nh, NeuralNet nn);

bool32 NeuralNetSave(NeuralNet nn, char *path);
bool32 NeuralNetLoad(Arena *arena, NeuralNet *nn, char *path, uint32 *layerSizes, uint32 layerCount);
void NeuralNetSizePushSingle(Arena *arena, NeuralNet *nn, uint32 size);
void NeuralNetSizePush(Arena *arena, NeuralNet *nn, uint32 *layerSizes, uint32 layerCount);
void NeuralNetCompile(Arena* arena, RandomSeries *series, NeuralNet *nn, uint32 *layerSizes, uint32 layerCount, bool32 randomize_params);

void NeuralNetForward(NeuralForward *nh, NeuralNet nn, Row x);
NeuralBack NeuralNetBackprop(Arena *arena, NeuralNet nn, Row x, Row y);
void NeuralNetUpdate(Arena *arena, NeuralNet nn, Matrix x_train, Matrix y_train, uint32 exampleCount, float32 rate);
void NeuralNetLearn(Arena *arena, RandomSeries *series, NeuralNet nn, Matrix x_train, Matrix y_train, uint32 epochs, float32 rate, uint32 batch_size);

