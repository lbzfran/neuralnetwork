
#include "random.h"
#define ARENA_IMPLEMENTATION
#include "arena.h"
#include <time.h>

int main(void)
{
    RandomSeries series = RandomSeed(time(0));
    Arena* arena = ArenaMalloc(Kilobytes(5));
    /**/
    /*NrlNet net = NrlNetArenaAlloc(arena, 2);*/
    /**/
    /*int layerSizes[2] = {*/
    /*    2, 2*/
    /*};*/
    /**/
    /*NrlNetInit(&series, arena, net, layerSizes);*/
    /*NrlNetPrint(net);*/
    printf("%f\n", RandomBilateral(&series));

    ArenaFree(arena);
    return 0;
}
