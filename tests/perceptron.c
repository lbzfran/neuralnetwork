

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
#define VarPrintFloatArray(x, k) do{ printf("%s = [\n", #x);\
                                  int j = 0;\
                                  for (int i = 0; i < ArrayCount(x); i++)\
                                  {\
                                      if (j++ % k == k - 1) { printf("\n"); }\
                                      printf("\t%f", x[i]);\
                                  }\
                                  printf("\n]\n");}while(0)

#define trainCount 8

int main(int argc, char **argv)
{
    RandomSeries series = {0};
    if (argc > 1)
    {
        series = RandomSeed((uint32)(argv[1] - '0'));
    }
    else
    {
        series = RandomSeed(69);
    }

    // sum of array
    float32 x[trainCount] = {0, 1, 1, 0, 1, 1, 1, 0};
    float32 w[trainCount];
    float32 b[trainCount];

    for (int32 i = 0; i < trainCount; i++)
    {
        w[i] = RandomBilateral(&series);
        b[i] = RandomBilateral(&series);
    }

    float32 sum = 0;
    for (int32 i = 0; i < trainCount; i++)
    {
        sum += x[i] * w[i] - b[i];
    }
    sum /= 4;
    // sum
    /*float32 x0 = 1;*/
    /*float32 w0 = RandomBilateral(&series);*/
    /*float32 b0 = RandomBilateral(&series);*/
    /**/
    /*float32 sum = x0 * w0 - b0;*/

    float32 y0 = sigmoidf(sum);

    /*VarPrintFloat(x0);*/
    /*VarPrintFloat(w0);*/
    /*VarPrintFloat(b0);*/
    /*VarPrintFloat(sum);*/
    /*VarPrintFloat(y0);*/

    VarPrintFloatArray(x, 6);
    VarPrintFloatArray(w, 6);
    VarPrintFloatArray(b, 6);
    VarPrintFloat(sum);
    VarPrintFloat(y0);

    return 0;
}
