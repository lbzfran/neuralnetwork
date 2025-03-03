

#include <math.h>
#include "random.h"
#define ARENA_IMPLEMENTATION
#include "arena.h"


float32 train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

float32
reluf(float32 x)
{
    return (x ? x > 0 : 0);
}

float32
linearf(float32 x)
{
    return (x > 0.5 ? 1 : 0);
}

float32
sigmoidf(float32 x)
{
    // NOTE(liam): return value between 0 and 1 based on how far along
    // the given float is from -inf to inf.
    return 1.f / (1.f + expf(-x));
}

#define VarPrintFloat(x) printf("%s = %f\n", (#x), x)
#define VarPrintFloatArray_(x, k) do{ printf("[\n");\
                                  int j = 0;\
                                  for (int i = 0; i < ArrayCount(x); i++)\
                                  {\
                                      if (j++ % k == k - 1) { printf("\n"); }\
                                      printf("\t%f", x[i]);\
                                  }\
                                  printf("\n]\n");}while(0)
#define VarPrintFloatArray(x, k) do{ printf("%s = ", #x); VarPrintFloatArray_(x, k); }while(0)
#define VarPrintFloatSubArray(x, i, k) do{ printf("%s[%d] = ", #x, i); VarPrintFloatArray_(x[i], k); }while(0)

#define trainCount 8

// currently assuming all layers are symmetrical in size
#define layerCount 8

int main(int argc, char **argv)
{
    RandomSeries series = {0};
    if (argc > 1)
    {
        RandomSeed(&series, (uint32)(argv[1] - '0'));
    }
    else
    {
        RandomSeed(&series, 69);
    }

    // sum of array
    float32 x[trainCount] = {0, 1, 1, 0, 1, 1, 1, 0};
    float32 w[layerCount][trainCount];
    float32 b[layerCount];
    float32 y[layerCount];

    for (uint32 l = 0; l < layerCount; l++)
    {
        b[l] = RandomBilateral(&series);
        for (uint32 i = 0; i < trainCount; i++)
        {
            w[l][i] = RandomBilateral(&series);
        }
    }

    /*for (int l = 0; l < layerCount; l++)*/
    /*{*/
    /*    VarPrintFloatArray(w[l], 6);*/
    /*    VarPrintFloatArray(b[l], 6);*/
    /*}*/

    // calculate a "layer" of neurons.
    for (int32 l = 0; l < layerCount; l++)
    {
        float32 sum = 0;
        for (int32 i = 0; i < trainCount; i++)
        {
            sum += x[i] * w[l][i];
        }
        sum += b[l];

        y[l] = sigmoidf(sum);
    }

    VarPrintFloatArray(x, 6);
    for (int l = 0; l < layerCount; l++)
    {
        VarPrintFloatSubArray(w, l, 6);
    }
    VarPrintFloatArray(b, 6);
    VarPrintFloatArray(y, 6);

    return 0;
}
