/*
 * ---------------
 * Liam Bagabag
 * Version: 1.0.0
 * require: none (inline)
 * ---------------
 */
#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>
//NOTE(liam): remove dependency on math.h
/*#include "types.h"*/

typedef struct vector2_t {
    float x, y;
} Vector2;

typedef struct vector3_t {
    float x, y, z;
} Vector3;

inline Vector2
Vector2Add(Vector2 a, Vector2 b)
{
    Vector2 res = {
        .x = a.x + b.x,
        .y = a.y + b.y
    };

    return(res);
}

inline Vector2
Vector2Sub(Vector2 a, Vector2 b)
{
    Vector2 res = {
        .x = a.x - b.x,
        .y = a.y - b.y
    };

    return(res);
}

inline Vector2
Vector2Mul(Vector2 a, float b)
{
    Vector2 res = {
        .x = a.x * b,
        .y = a.y * b
    };

    return(res);
}

inline float
Vector2Dot(Vector2 a, Vector2 b)
{
    float res = (a.x * b.x) + (a.y * b.y);

    return(res);
}

inline float
Vector2Cross(Vector2 a, Vector2 b)
{
    float res = (a.x * b.y) - (a.y * b.x);

    return(res);
}

inline float
Vector2Magnitude(Vector2 a)
{
    float res = sqrt(a.x * a.x + a.y * a.y);

    return(res);
}

inline Vector3
Vector3Add(Vector3 a, Vector3 b)
{
    Vector3 res = {
        .x = a.x + b.x,
        .y = a.y + b.y,
        .z = a.z + b.z
    };

    return(res);
}

inline Vector3
Vector3Sub(Vector3 a, Vector3 b)
{
    Vector3 res = {
        .x = a.x - b.x,
        .y = a.y - b.y,
        .z = a.z - b.z
    };

    return(res);
}

inline Vector3
Vector3Mul(Vector3 a, float b)
{
    Vector3 res = {
        .x = a.x * b,
        .y = a.y * b,
        .z = a.z * b
    };

    return(res);
}

inline float
Vector3Dot(Vector3 a, Vector3 b)
{
    float res = (a.x * b.x) + (a.y * b.y) + (a.z * b.z);

    return(res);
}

inline float
Vector3Cross(Vector3 a, Vector3 b)
{
    float res = (a.y * b.z - a.z * b.y) + (a.z * b.x - a.x * b.z) + (a.x * b.y - a.y * b.x);

    return(res);
}

inline float
Vector3Magnitude(Vector3 a)
{
    float res = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);

    return(res);
}

#endif
