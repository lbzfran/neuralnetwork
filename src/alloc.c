
#include <string.h>
#include "alloc.h"
#include "base.h"

// NOTE(liam): does not inherently allocate memory.
// Must be performed as user sees fit before call.
void
ArenaAlloc(Arena* arena, memory_index size, uint8* base_addr)
{
    arena->size = size;
    arena->pos = 0;
    arena->base = base_addr;
    if (!base_addr) {
        // NOTE(liam): if NULL, auto-generate memory using malloc.
        arena->base = a_malloc(size);
    }
    arena->tempCount = 0;
}

// NOTE(liam): convenience arena wrapper that uses
// malloc implementation defined by user (or stdlib).
Arena*
ArenaMalloc(memory_index size)
{
    Arena* arena = (Arena*)a_malloc(size);
    ArenaAlloc(arena, size, (uint8*)malloc(size));

    return(arena);
}

// NOTE(liam): this can be potentially skipped if you're
// using other forms of memory allocation (VirtualAlloc, mmap).
// A good way to check if you need it is if you used the
// ArenaMalloc function for allocation.
void
ArenaFree(Arena* arena)
{
    if (arena) {
        a_free(arena->base);
        a_free(arena);
    }
}

void*
ArenaPush(Arena* arena, memory_index size)
{
    Assert(arena->pos + size < arena->size, "requested alloc size exceeds arena size.")
    void* res = arena->base + arena->pos;
    arena->pos += size;

    return(res);
}

void*
ArenaPushZero(Arena* arena, memory_index size)
{
    Assert(arena->pos + size < arena->size, "requested alloc size exceeds arena size.")
    void* res = arena->base + arena->pos;
    arena->pos += size;

    memset(res, 0, size);

    return(res);
}

//TODO(liam): finish implementation
/*void ArenaPop(Arena *arena, memory_index size) {*/
/*}*/

uint64
ArenaGetPos(Arena *arena)
{
    return(arena->pos);
}

void
ArenaSetPos(Arena *arena, memory_index pos)
{
    Assert(pos <= arena->pos, "Setting position beyond current arena allocation.");
    arena->pos = pos;
}

// NOTE(liam): effectively resets the Arena.
void ArenaClear(Arena *arena)
{
    arena->pos = 0;
    arena->tempCount = 0;
}

// NOTE(liam): temporary memory.
ArenaTemp
ArenaTempBegin(Arena *arena)
{
    ArenaTemp res;

    res.arena = arena;
    res.pos = arena->pos;

    arena->tempCount++;

    return(res);
}

void
ArenaTempEnd(ArenaTemp temp)
{
    Arena* arena = temp.arena;

    Assert(arena->pos >= temp.pos, "Arena position is less than temporary memory's position. Likely user-coded error.");
    arena->pos = temp.pos;

    Assert(arena->tempCount > 0, "Attempt to decrement Arena's temporary memory count when it is already 0.");
    arena->tempCount--;
}

// NOTE(liam): should call after temp use.
// need to make sure all temps are accounted for before
// resuming allocations.
void
ArenaTempCheck(Arena* arena)
{
    Assert(arena->tempCount == 0, "")
}

ArenaTemp
GetScratch(Arena* arena)
{
    Assert(arena->pos + sizeof(ArenaTemp) <= arena->size, "requested temp alloc exceeds arena size.");

    ArenaTemp temp = ArenaTempBegin(arena);
    return temp;
}
