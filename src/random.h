/*
 * ---------------
 * Liam Bagabag
 * Version: 2.0.0
 * requires: n/a
 * ---------------
 */
#ifndef RANDOM_H
#define RANDOM_H

#include "def.h"
#include "arena.h"

typedef struct random_series {
    uint32 index;
} RandomSeries;

void RandomSeed(RandomSeries* series, uint32 value);
uint32 RandomNextInt(RandomSeries* series);
uint32 RandomChoice(RandomSeries* series, uint32 N); // [0, N)
float32 RandomUnilateral(RandomSeries* series); // [0,1]
float32 RandomBilateral(RandomSeries* series); // [-1,1]
float32 RandomBetween(RandomSeries* series, float32 min, float32 max); // (min, max)
#endif //RANDOM_H
