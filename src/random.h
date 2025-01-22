#ifndef RANDOM_H
#define RANDOM_H

#include "base.h"

typedef struct random_series {
    unsigned int index;
} RandomSeries;

RandomSeries RandomSeed(uint value);
uint RandomNextInt(RandomSeries* series);
uint RandomChoice(RandomSeries* series, uint N);
float RandomUnilateral(RandomSeries* series);
float RandomBilateral(RandomSeries* series);
float RandomLerp(float A, float t, float B);
float RandomBetween(RandomSeries* series, float min, float max);

#endif

#ifdef RAND_IMPLEMENTATION

# define MaxRandomNumber 0xF9B4C0
# define MinRandomNumber 0x02B06C

//TODO(liam): find a formula to generate this table later on.
static int RandomNumberTable[] = {
    0x3F1A2B, 0xA1D5C3, 0x6D9B84, 0x9C7E21, 0xE7465F, 0x4290FA, 0xF4A832,
    0x1B7C5E, 0xD97A3B, 0x8A5D67, 0xC33B1F, 0x5E22D4, 0x710F56, 0x3D4A61,
    0xB2435C, 0x14C9A0, 0x6B35D8, 0xF889A2, 0x9E6F44, 0x2A7F1C, 0x7C8920,
    0x5723C7, 0x304B59, 0x46A1FD, 0x207D14, 0x8C3D70, 0x1398C4, 0x6F2A89,
    0xD55231, 0x92F6B7, 0x40C9A4, 0xB3E702, 0x5D9A81, 0x1A6E9F, 0xA98D34,
    0x64F76B, 0x72708B, 0xF2C11E, 0xB6C941, 0x15D3F4, 0x883DF2, 0x29A571,
    0x04B269, 0x39A30E, 0xA256F3, 0xF76D21, 0x51D4E2, 0x0B192A, 0x987543,
    0x6A0C82, 0x3C7129, 0x17A4E7, 0xE4625B, 0x774C91, 0x7A5298, 0x8F3645,
    0x0A1D84, 0xD0E8B3, 0x41F5B9, 0x5AC7D6, 0xC229F0, 0x1167A0, 0x573C89,
    0x9335BD, 0x254F6D, 0x5C8172, 0x373E8A, 0xE9334F, 0x19170D, 0x1206F9,
    0xB8378C, 0x70A452, 0x28C8F5, 0xC1B6A1, 0x996841, 0x33D0F7, 0x562B90,
    0xA473E2, 0x73B150, 0xF9B4C0, 0x45D2A1, 0x02B06C, 0xB02E7F, 0x85D3A7,
    0x9D0F6E, 0x84F7A8, 0xD49E21, 0x1C7B5A, 0x22A4F3, 0xC49E58, 0x787D3B,
    0xE64F92, 0x91C79F, 0xD8682D, 0x630C41, 0x097F52, 0x60E7B6, 0x79F52D,
    0x8239A7, 0x4E634F, 0x31F2C0, 0xB59F6A, 0x6708F7, 0x101B3D, 0x44E0A9,
    0x23415E, 0x663D28, 0x16843F, 0x089D62, 0x50C8D7, 0xC8A6F9, 0xA7248C,
    0x26F5D1, 0x9B0F30, 0x0F6C75, 0x81B95A, 0x3B8C7D, 0x90A51E, 0x2490B4,
    0x21F60A, 0xF3A792, 0x74D219, 0x94B1F3, 0x7E625D, 0x61D09B, 0x65C2D0,
    0x4F3A25, 0x97F01B, 0x5B7846, 0xD4739C, 0x2B1D64, 0xF50192, 0xD63784,
    0xAC9C01, 0x7382F5, 0x650FED, 0x1FB964, 0x52A601, 0x98C6FB, 0x0D4912,
    0x0E567A, 0x8F2B3C, 0x6A4907, 0x58A29D, 0x70237B, 0x49B276, 0x3A1F94,
    0xB55D8D, 0xCF6C82, 0x61F437, 0x09D0F1, 0x18B3E9, 0x58D9A3, 0x71CE04,
    0x7F8160, 0x1C89F8, 0x3D3AB0, 0x9C62C5, 0xDFF50F, 0x87C9DA, 0x7BC44A,
    0x02B346, 0xDA30F5, 0x1E579B, 0xCC7F18, 0xB2C107, 0x76F81A, 0x6E243F,
    0x39825C, 0x1140E7, 0x503264, 0xDFB681, 0xEE415F, 0xD0B8A2, 0x2A61F5,
    0xBE5D19, 0x14D7B3, 0x75F0C2, 0x5439E6, 0x8B724C, 0x6E0913, 0xF183BC,
    0x7A9D8E, 0xAF084C, 0x88D23E, 0x4A1759, 0x9B6281, 0xC682C8, 0x1D8032,
    0x5A291F, 0x4F49A7, 0xA4B68F, 0x763D40, 0xD213A6, 0x8245FD, 0xF601E3
};

inline RandomSeries
RandomSeed(uint value)
{
    RandomSeries series;
    series.index = (value % ArrayCount(RandomNumberTable));
    return(series);
}

inline uint
RandomNextInt(RandomSeries* series)
{
    // returns current value of index, and increments index position.
    uint res = RandomNumberTable[series->index++];
    if (series->index == ArrayCount(RandomNumberTable)) {
        series->index = 0;
    }
    return(res);
}

inline uint
RandomChoice(RandomSeries* series, uint N)
{
    // random choice between [0, N).

    uint res = (RandomNextInt(series) % N);

    return(res);
}

inline float
RandomUnilateral(RandomSeries* series)
{
    // range of [0 to 1].
    float div = 1.0f / (float)MaxRandomNumber;
    float res = div * (float)RandomNextInt(series);

    return(res);
}

inline float
RandomBilateral(RandomSeries* series)
{
    // range of [-1 to 1].
    float res = 2.0f * RandomUnilateral(series) - 1.0f;

    return(res);
}

inline float
RandomLerp(float A, float t, float B)
{
    // Linear Interpolation
    float res = (1.0f - t) * A + t * B;

    return(res);
}

inline float
RandomBetween(RandomSeries* series, float min, float max)
{
    /*float range = max - min;*/
    /*float res   = min + RandomUnilateral(series) * range;*/
    float res = RandomLerp(min, RandomUnilateral(series), max);

    return(res);
}

#endif //RANDOM_H
