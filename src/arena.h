/*
 * ---------------
 * Liam Bagabag
 * Version: a2.0
 * Requires: n/a
 * ---------------
 */
#ifndef ARENA_H
#define ARENA_H

// NOTE(liam): define 'ARENA_USEMALLOC' if you want to allocate memory yourself.
// afterwards, you have to define both 'a_alloc' and 'a_free', or
// it will fallback to using the std lib malloc implementation.
// NOTE(liam): when not defining 'ARENA_USEMALLOC', you will need to compile
// the respective c file (arena_memory_<linux/win32>.c) with your project,
// or define the respective functions yourself.
/*#define ARENA_USEMALLOC*/

#include "def.h"

#define DEFAULT_ALIGNMENT sizeof(void*)
#define DEFAULT_BLOCKSIZE Kilobytes(16)

// ALIGNMENT START (REQUIRES C11 ver)
#if __STDC_VERSION__ < 201112L && (defined(COMPILER_GCC) || defined(COMPILER_TCC))
# define _Alignas(t) __attribute__((__aligned__(t)))
# define _Alignof(t) __alignof__(t)
#endif

#define alignas _Alignas
#define alignof _Alignof
// ALIGNMENT END

typedef struct memory_arena_footer {
    uint8* base;
    memory_index size;
    memory_index pos;
    memory_index padding;
} ArenaFooter;

typedef struct memory_arena {
    uint8* base;
    memory_index size;
    memory_index pos; // aka used memory idx

    memory_index minimumBlockSize;

    uint32 blockCount;
    uint32 tempCount;
} Arena;

typedef struct memory_arena_temp {
    Arena* arena;
    uint8* base;
    memory_index pos;
    memory_index padding;
} ArenaTemp;

void* ArenaPush(Arena*, memory_index, memory_index);
void ArenaPop(Arena* arena, memory_index size);
void* ArenaCopy(memory_index, void*, void*);
ArenaFooter* GetFooter(Arena* arena);

memory_index ArenaGetEffectiveSize(Arena* arena, memory_index sizeInit, memory_index alignment);
memory_index ArenaGetAlignmentOffset(Arena* arena, memory_index alignment);
memory_index ArenaGetRemainingSize(Arena* arena, memory_index alignment);

void ArenaFreeCurrentBlock(Arena* arena);

// NOTE(liam): auto-aligned Push Instructions.
#define PushArray(arena, t, c) (t*)ArenaPush((arena),sizeof(t)*(c), alignof(t))
#define PushStruct(arena, t) PushArray(arena, t, 1)
#define PushSize(arena, s) ArenaPush((arena), (s), alignof(s))
#define PushCopy(arena, s, src) ArenaCopy(s, src, ArenaPush(arena, s, alignof(s)))

// NOTE(liam): Set Alignment Manually.
#define PushArrayAlign(arena, t, c, ...) (t*)ArenaPush((arena),sizeof(t)*(c), ## __VA_ARGS__)
#define PushStructAlign(arena, t, ...) PushArray(arena, t, ## __VA_ARGS__)
#define PushSizeAlign(arena, s, ...) ArenaPush((arena), (s), ## __VA_ARGS__)
#define PushCopyAlign(arena, s, src, ...) ArenaCopy(s, src, ArenaPush(arena, s, ## __VA_ARGS__))
void ArenaFillZero(memory_index size, void *ptr);

uint64 ArenaGetPos(Arena*);

void ArenaSetMinimumBlockSize(Arena* arena, memory_index minimumBlockSize);
void ArenaSetPos(Arena*, memory_index);
void ArenaClear(Arena*);
#define ArenaFree(arena) ArenaClear(arena);

void SubArena(Arena* subArena, Arena* arena, memory_index size, memory_index alignment);

ArenaTemp ArenaTempBegin(Arena*);
void ArenaTempEnd(ArenaTemp);
void ArenaTempCheck(Arena*);

ArenaTemp ArenaScratchCreate(Arena*);
#define ArenaScratchFree(t) ArenaTempEnd(t)

#define ZeroStruct(in) ArenaFillZero(sizeof(in), &(in))
#define ZeroArray(n, ptr) ArenaFillZero((n)*sizeof((ptr)[0]), (ptr))

#endif //ARENA_H

#ifdef ARENA_IMPLEMENTATION

# ifdef ARENA_USEMALLOC
#  include <stdlib.h>
#  define AllocateMemory malloc
#  define DeallocateMemory free
# else
#include <sys/mman.h>
static void* AllocateMemory(memory_index size)
{
    void* res = /*(memory_block*)*/
        mmap(NULL, size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (res == MAP_FAILED)
    {
        perror("Failed Allocation.");
        return NULL;
    }
    return(res);
}

static void DeallocateMemory(void* ptr, memory_index size)
{
    bool32 res = munmap(ptr, size);
    if (res == -1)
    {
        perror("Failed Deallocation.");
    }
}
# endif

void
ArenaFillZero(memory_index size, void *ptr) // effectively memcpy
{
    uint8* byte = (uint8*) ptr;
    while (size--) {
        *byte++ = 0;
    }
}

void
ArenaSetMinimumBlockSize(Arena* arena, memory_index minimumBlockSize)
{
    arena->minimumBlockSize = minimumBlockSize;
}

memory_index
ArenaGetAlignmentOffset(Arena* arena, memory_index alignment)
{
    memory_index alignmentOffset = 0;

    memory_index resPointer = (memory_index)arena->base + arena->pos;
    memory_index alignmentMask = alignment - 1;
    if (resPointer & alignmentMask)
    {
        alignmentOffset = alignment - (resPointer & alignmentMask);
    }

    return(alignmentOffset);
}

memory_index
ArenaGetRemainingSize(Arena* arena, memory_index alignment)
{
    memory_index res = arena->size - (arena->pos + ArenaGetAlignmentOffset(arena, alignment));
    return(res);
}

memory_index
ArenaGetEffectiveSize(Arena* arena, memory_index sizeInit, memory_index alignment)
{
    memory_index size = sizeInit;

    memory_index alignmentOffset = ArenaGetAlignmentOffset(arena, alignment);
    size += alignmentOffset;

    return(size);
}

bool32
ArenaCanStoreSize(Arena* arena, memory_index sizeInit, memory_index alignment)
{
    if (!alignment) alignment = DEFAULT_ALIGNMENT;

    memory_index size = ArenaGetEffectiveSize(arena, sizeInit, alignment);
    bool32 res = (arena->pos + size <= arena->size);

    return(res);
}

ArenaFooter*
GetFooter(Arena* arena)
{
    ArenaFooter *res = (ArenaFooter*)(arena->base + arena->size);

    return(res);
}

void*
ArenaPush(Arena* arena, memory_index sizeInit, memory_index alignment)
{
    if (!alignment) alignment = DEFAULT_ALIGNMENT;

    //NOTE(liam): rounds allocation up to set align properly.
    memory_index size = ArenaGetEffectiveSize(arena, sizeInit, alignment);

    /*Assert(arena->pos + size < arena->size, "requested alloc size exceeds arena size.")*/
    if ((arena->pos + size) > arena->size)
    {
        // NOTE(liam): if min block size was never set, set it.
        if (!arena->minimumBlockSize)
        {
            // TODO(liam): tune block sizing
            arena->minimumBlockSize = DEFAULT_BLOCKSIZE; // 1024 * 1024
        }

        ArenaFooter save = {0};
        save.base = arena->base;
        save.size = arena->size;
        save.pos = arena->pos;

        // NOTE(liam): base should automatically align after allocating again.
        size = sizeInit;
        memory_index blockSize = Max(size + sizeof(struct memory_arena_footer), arena->minimumBlockSize);
        arena->size = blockSize - sizeof(struct memory_arena_footer);
        arena->base = (uint8*)AllocateMemory(blockSize);
        arena->pos = 0;
        arena->blockCount++;

        ArenaFooter* footer = GetFooter(arena);
        *footer = save;
    }
    Assert(((arena->pos + size) <= arena->size) && "new allocation of dynamic arena somehow failed...");

    memory_index alignmentOffset = ArenaGetAlignmentOffset(arena, alignment);
    void* res = (void*)(arena->base + arena->pos - alignmentOffset);
    arena->pos += size;

    Assert((size >= sizeInit) && "requested alloc exceeds arena size after alignment.");

    return(res);
}

void*
ArenaCopy(memory_index size, void* src, void* dst)
{
    uint8* srcPos = (uint8*)src;
    uint8* dstPos = (uint8*)dst;
    while (size--)
    {
        *dstPos++ = *srcPos++;
    }
    return(dst);
}

void
SubArena(Arena* subArena, Arena* arena, memory_index size, memory_index alignment)
{
    if (!alignment) alignment = DEFAULT_ALIGNMENT;

    subArena->size = size;
    subArena->base = (uint8*)PushSizeAlign(arena, size, alignment);
    subArena->pos = 0;
    subArena->tempCount = 0;
}


void
ArenaPop(Arena* arena, memory_index size)
{
    if ((arena->size - size) > 0)
    {
        arena->size -= size;
    }
    else { arena->size = 0; }
}

// NOTE(liam): effectively resets the Arena.
void
ArenaClear(Arena *arena)
{
    while (arena->blockCount)
    {
        ArenaFreeCurrentBlock(arena);
    }
}

// NOTE(liam): temporary memory.
ArenaTemp
ArenaTempBegin(Arena *arena)
{
    ArenaTemp res;

    res.arena = arena;
    res.base = arena->base;
    res.pos = arena->pos;

    arena->tempCount++;

    return(res);
}

void
ArenaFreeCurrentBlock(Arena* arena)
{
    void* freedBlock = arena->base;
    memory_index freedBlockSize = arena->size;

    ArenaFooter* footer = GetFooter(arena);

    arena->base = footer->base;
    arena->size = footer->size;
    arena->pos  = footer->pos;

    DeallocateMemory(freedBlock, freedBlockSize);

    arena->blockCount--;
}

void
ArenaTempEnd(ArenaTemp temp)
{
    Arena* arena = temp.arena;

    while(arena->base != temp.base)
    {
        ArenaFreeCurrentBlock(arena);
    }

    Assert((arena->pos >= temp.pos) && "Arena position is less than temporary memory's position. Likely user-coded error.");
    arena->pos = temp.pos;

    Assert((arena->tempCount > 0) && "Attempt to decrement Arena's temporary memory count when it is already 0.");
    arena->tempCount--;
}

// NOTE(liam): should call after finishing temp use.
// need to make sure all temps are accounted for before
// resuming allocations.
void
ArenaTempCheck(Arena* arena)
{
    Assert(arena->tempCount == 0);
}

ArenaTemp
ArenaScratchCreate(Arena* arena)
{
    //TODO(liam): replace assertion.
    /*Assert((arena->pos + sizeof(ArenaTemp) <= arena->size) && "requested temp alloc exceeds arena size.");*/

    ArenaTemp temp = ArenaTempBegin(arena);
    return temp;
}

#endif // ARENA_IMPLEMENTATION
