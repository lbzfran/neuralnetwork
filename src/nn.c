
#include <stdio.h>
#define ZRAND_IMPLEMENTATION
#include "random.h"

static RandomSeries local_series;

int main(void)
{
    local_series = RandomSeed(69);
    return 0;
}
