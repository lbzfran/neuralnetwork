
#include "random.h"

static const uint32 A = 1664525;
static const uint32 C = 1013904223;
static const uint32 M = 4294967295;

inline static float32
Lerp(float32 a, float32 t, float32 b)
{
    float32 res = a + (b - a) * t;

    return(res);
}

inline void
RandomSeed(RandomSeries *series, uint32 value)
{
    series->index = (A * value + C) % M;
}

inline uint32
RandomNextInt(RandomSeries *series)
{
    // returns current value of index, and increments index position.
    /*uint32 res = (uint32)(series->index/65536);*/
    uint32 res = series->index;
    series->index = (A * series->index + C) % M;

    return(res);
}

inline uint32
RandomChoice(RandomSeries *series, uint32 N)
{
    // random choice between [0, N).

    uint32 res = (RandomNextInt(series) % N);

    return(res);
}

inline float32
RandomUnilateral(RandomSeries *series)
{
    // range of [0 to 1].
    float32 div = 1.0f / M;
    float32 res = div * (float32)RandomNextInt(series);

    return(res);
}

inline float32
RandomBilateral(RandomSeries* series)
{
    // range of [-1 to 1].
    float32 res = 2.0f * RandomUnilateral(series) - 1.0f;

    return(res);
}

inline float32
RandomBetween(RandomSeries* series, float32 min, float32 max)
{
    /*float32 range = max - min;*/
    /*float32 res   = min + RandomUnilateral(series) * range;*/
    float32 res = Lerp(min, RandomUnilateral(series), max);

    return(res);
}

