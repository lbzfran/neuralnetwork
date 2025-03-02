

#define MATRIX_IMPLEMENTATION
#include "../src/matrix.h"

#define RAND_IMPLEMENTATION
#include "../src/random.h"

#define ARENA_IMPLEMENTATION
#include "../src/arena.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static RandomSeries local_series;

/*typedef float[3] sample;*/

/*sample or_train[] = {*/
/*    {0, 0, 0},*/
/*    {0, 1, 1},*/
/*    {1, 0, 1},*/
/*    {1, 1, 1}*/
/*};*/


float *train[3];
#define trainCount (sizeof(train)/sizeof(train[0]))

float cost(float w1, float w2, float b)
{
    float result = 0.0f;
    for (size_t i = 0; i < trainCount; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + b);
        float d = y - train[i][2];
        result += d * d;
    }
    result /= trainCount;
    return result;
}

int main(void)
{
    Arena* local_arena = ArenaMalloc(Kilobytes(2));
    local_series = RandomSeed(time(0));

    float w1 = RandomBilateral(&local_series) * 10.f;
    float w2 = RandomBilateral(&local_series) * 10.f;

    float b = RandomBilateral(&local_series) * 5.f;
    float eps = 1e-1;
    float rate = 1e-1;

    for (size_t i = 0; i < 20000; i++) {
        float c = cost(w1, w2, b);
        float dw1 = (cost(w1 + eps, w2, b) - c)/eps;
        float dw2 = (cost(w1, w2 + eps, b) - c)/eps;
        float db =  (cost(w1, w2, b + eps) - c)/eps;

        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
        printf("w1 = %f  w2 = %f b = %f cost = %f\n", w1, w2, b, c);
    }

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%zu | %zu = %f\n", i, j, sigmoidf(i * w1 + j * w2 + b));
        }
    }

    ArenaFree(local_arena);
    return 0;
}
