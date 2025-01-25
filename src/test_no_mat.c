

#define ARENA_IMPLEMENTATION
#include "arena.h"
#include "random.h"

#include <stdio.h>
#include <stdlib.h>

static RandomSeries local_series;

float train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};
#define trainCount (sizeof(train)/sizeof(train[0]))

float cost(float w, float b)
{
    float result = 0.0f;
    for (size_t i = 0; i < trainCount; i++) {
        float x = train[i][0];
        float y = x * w + b;
        float d = y - train[i][1];
        result += d * d;
    }
    result /= trainCount;
    return result;
}

int main(void)
{
    Arena* local_arena = ArenaMalloc(Kilobytes(2));
    local_series = RandomSeed(69);

    float w = RandomBilateral(&local_series) * 10.f;
    float b = RandomBilateral(&local_series) * 5.f;
    float eps = 1e-3;
    float rate = 1e-3;

    printf("%f\n", cost(w, b));
    for (size_t i = 0; i < 500; i++) {
        float c = cost(w, b);
        float dw = (cost(w + eps, b) - c)/eps;
        float db = (cost(w, b + eps) - c)/eps;

        w -= rate * dw;
        b -= rate * db;

        printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
    }
    printf("----------------------\n");
    printf("w = %f, b = %f\n", w, b);

    ArenaFree(local_arena);
    return 0;
}
