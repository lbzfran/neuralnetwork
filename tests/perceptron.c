

#include <math.h>
#include "random.h"
#define ARENA_IMPLEMENTATION
#include "arena.h"

// NOTE(liam): does not handle new data well.
// train data is too limited; severe overfitting.

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

// NOTE(liam): derivative of sigmoidf
float32
dsigmoidf(float32 z)
{
    return z * (1 - z);
}

float32
squaref(float32 x)
{
    return x * x;
}

// NOTE(liam): calculate one given an output of a full cycle of training
// whose size matches an established label data.
float32 cost(float32 *y_pred, float32 *y, uint32 size)
{
    float32 sum = 0;
    for (uint32 i = 0; i < size; i++)
    {
        float32 y_error = *(y_pred + i) - *(y + i);
        sum += squaref(y_error);
    }

    return sum / size;
}

// NOTE(liam):
// chain rule:
// dc[0] / dw[L] = dz[L] / dw[L] * da[L] / dz[L] * dc[0] / da[L]
//
// params:
//  a[L-1] * f'(z[L]) * 2(a[L] - y)
//  cost
//
/*float32 backprop(*/
/*    float32 *x, // size l-1*/
/*    float32 **w, // size l*/
/*    float32 *b_l, // size 1*/
/*    float rate,*/
/*    float c,*/
/*    uint32 size*/
/*)*/
/*{*/
/*    float32 gradient = c * dsigmoidf(z);*/
/*    for (uint32 i = 0; i < size; i++)*/
/*    {*/
/*        w[i] += rate * gradient * x[i];*/
/*    }*/
/*    *b_l += rate * gradient;*/
/*}*/

#define VarPrintFloat(x) printf("%s = %f\n", (#x), x)
#define VarPrintFloatArray_(x, k) do{ printf("[\n");\
                                  int j = 0;\
                                  int s = ArrayCount(x);\
                                  for (int i = 0; i < s; i++)\
                                  {\
                                      printf("\t%f", x[i]);\
                                      if (j++ % k == k - 1 && i + 1 < s) { printf("\n"); }\
                                  }\
                                  printf("\n]\n");}while(0)
#define VarPrintFloatArray(x, k) do{ printf("%s = ", #x); VarPrintFloatArray_(x, k); }while(0)
#define VarPrintFloatSubArray(x, i, k) do{ printf("%s[%d] = ", #x, i); VarPrintFloatArray_(x[i], k); }while(0)

#define trainCount 8

// currently assuming all layers are symmetrical in size
#define layerCount 4

int main(int argc, char **argv)
{
    RandomSeries series = {0};
    uint32 seed = 69;

    if (argc > 1)
    {
        sscanf(argv[1], "%u", &seed);
    }

    RandomSeed(&series, seed);
    printf("seed: %d\n", seed);

    float32 rate = 0.01;
    float32 y_error = 1.f;
    // sum of array
    float32 x_train[trainCount] = {0, 0, 0, 1, 1, 0, 1, 1};
    float32 x_test[trainCount] = {0, 0, 0, 0, 0, 0, 0, 0};
    float32 w[layerCount][trainCount];
    float32 b[layerCount];
    float32 a[layerCount];
    float32 y[layerCount] = {0, 1, 1, 1};
    float32 y_test[layerCount] = {0, 0, 0, 0};

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
    /*    printf("%f\n", RandomUnilateral(&series));*/
    /*}*/

    // forward propagation
    // calculate a "layer" of neurons.
    VarPrintFloatArray(x_train, 2);
    VarPrintFloatArray(y, 4);

    uint32 l_iter = 0;
    while (y_error > 0.00010)
    {
        for (int32 l = 0; l < layerCount; l++)
        {
            float32 z = 0;
            for (int32 i = 0; i < trainCount; i++)
            {
                z += x_train[i] * w[l][i];
            }
            z += b[l];

            a[l] = sigmoidf(z);
        }
        y_error = cost(a, y, layerCount);

        for (uint32 l = 0; l < layerCount; l++)
        {
            float32 d_error = 2 * (a[l] - y[l]) / layerCount;
            float32 gradient =  d_error * dsigmoidf(a[l]);
            for (uint32 i = 0; i < trainCount; i++)
            {
                w[l][i] -= rate * gradient * x_train[i];
            }
            b[l] -= rate * gradient;
        }
        VarPrintFloat(y_error);
    }

    for (int l = 0; l < layerCount; l++)
    {
        VarPrintFloatSubArray(w, l, 2);
    }
    VarPrintFloatArray(b, 4);
    VarPrintFloatArray(a, 4);
    VarPrintFloat(y_error);

    return 0;
}
