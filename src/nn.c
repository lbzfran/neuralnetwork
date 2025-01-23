
#include "random.h"
#define MATRIX_IMPLEMENTATION
#define ALLOC_IMPLEMENTATION
#define RAND_IMPLEMENTATION

#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

static RandomSeries local_series;

float train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};

int main(void)
{
    Arena* local_arena = ArenaMalloc(Kilobytes(2));
    local_series = RandomSeed(69);

    float w = RandomBilateral(&local_series);


    ArenaFree(local_arena);
    return 0;
}
