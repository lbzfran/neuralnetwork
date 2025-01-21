#ifndef ALLOC_H
#define ALLOC_H

// PLATFORM-INDEPENDENT
#include "base.h"

// DEFINE YOUR OWN MALLOC HERE.
#include <stdlib.h>
#define a_malloc malloc
#define a_realloc realloc
#define a_free free

typedef struct memory_arena {
    memory_index size;
    uint8* base;
    memory_index pos; // aka used memory idx

    uint32 tempCount;
} Arena;

typedef struct memory_arena_temp {
    Arena* arena;
    memory_index pos;
} ArenaTemp;

void ArenaAlloc(Arena*, memory_index, uint8*);
Arena* ArenaMalloc(memory_index size);
void ArenaFree(Arena*);

void* ArenaPush(Arena*, memory_index);
void* ArenaPushZero(Arena*, memory_index);

// NOTE(liam): helper macros
#define PushArray(arena, t, c) (t*)ArenaPush((arena),sizeof(t)*(c))
#define PushStruct(arena, t) PushArray(arena, t, 1)
#define PushArrayZero(arena, t, c) (t*)ArenaPushZero((arena),sizeof(t)*(c))
#define PushStructZero(arena, t) PushArrayZero(arena, t, 1)

void ArenaPop(Arena*, memory_index);
uint64 ArenaGetPos(Arena*);

void ArenaSetPos(Arena*, memory_index);
void ArenaClear(Arena*);

ArenaTemp ArenaTempBegin(Arena*); // grabs arena's position
void ArenaTempEnd(ArenaTemp);     // restores arena's position
void ArenaTempCheck(Arena*);

ArenaTemp GetScratch(Arena*);
#define FreeScratch(t) ArenaTempEnd(t)

#endif
