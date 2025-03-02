
#define MATRIX_IMPLEMENTATION
#include "../src/matrix.h"

#define ARENA_IMPLEMENTATION
#include "../src/arena.h"

#define RAND_IMPLEMENTATION
#include "../src/random.h"

static RandomSeries series;

int main(void)
{
    Arena* arena = ArenaMalloc(Kilobytes(1));
    series = RandomSeed(1020);

    Matrix m = MatrixArenaAlloc(arena, 4, 4);
    Matrix w = MatrixArenaAlloc(arena, 4, 4);

    MatrixFill(w, 1);
    MatrixPrint(w);

    MatrixRandomize(&series, m, -10, 10);
    MatrixPrint(m);

    MatrixSigmoid(m);
    MatrixPrint(m);

    ArenaFree(arena);
    return 0;
}
