#ifndef BASE_H
#define BASE_H

// NOTE(liam): context cracking
// used for context-specific settings
# if defined(_WIN32) || defined(_WIN64)
#  define OS_WINDOWS 1
# elif defined(__linux__) || defined(__gnu_linux__)
#  define OS_LINUX 1
# elif defined(__APPLE__) && defined(__MACH__)
# define OS_MAC 1
# else
# error "Unsupported Operating System."
# endif

# if defined(__M_AMD64) || defined(__amd64__)
#  define ARCH_X64 1
# elif defined(__M_I86) || defined(__i386__)
#  define ARCH_X86 1
# elif defined(__M_ARM) || defined(__arm__)
#  define ARCH_ARM 1
# elif defined(__aarch64__)
#  define ARCH_ARM64 1
# else
#  error "Unsupported Architecture."
# endif

# ifdef __clang__
#  define COMPILER_CLANG 1
# elif __GNUC__
#  define COMPILER_GCC 1
# else
#  error "Unsupported Compiler."
# endif

# if defined(OS_WINDOWS)
#  define CLEAR system("cls")
# elif defined(OS_LINUX) || defined(OS_MAC)
#  define CLEAR system("clear")
# else
#  define CLEAR 0
# endif

// aliases

# define global     static
# define local      static
# define function   static

// some funny redefs
# define AND &&
# define OR ||
# define NOT !
# define elif else if

// NOTE(liam): specific type sizes
# include <stdio.h>
# include <stdint.h>

typedef uint8_t     uint8;  // unsigned char
typedef uint16_t    uint16; // unsigned int
typedef uint32_t    uint32; // unsigned long int
typedef uint64_t    uint64; // unsigned long long int

typedef int8_t      int8;   // signed char
typedef int16_t     int16;  // signed int
typedef int32_t     int32;  // signed long int
typedef int64_t     int64;  // signed long long int

typedef float       real32;
typedef double      real64;
typedef int32       bool32;
typedef size_t memory_index;

typedef void VoidFunc(void);

int ap_int(int a, int b);
int ap_float(float a, float b);

// NOTE(liam): EXPERIMENTAL: here for demo purposes
# define OVERLOADING
# ifdef  OVERLOADING
#  define ap(a, b) _Generic((a),    \
    int     :   ap_int((a), (b)),   \
    float   :   ap_float((a), (b))  \
)
# endif

// NOTE(liam): general macros
# define Swap(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
# define Min(a, b)  ((a) < (b) ? (a) : (b))
# define Max(a, b)  ((a) > (b) ? (a) : (b))

# define ArrayCount(a) (sizeof(a)/sizeof(*(a)))

# define Kilobytes(V) ((V)*1024LL)
# define Megabytes(V) (Kilobytes(V)*1024LL)
# define Gigabytes(V) (Megabytes(V)*1024LL)
# define Terabytes(V) (Gigabytes(V)*1024LL)

#include <stdlib.h>
// DEBUG START
# define ENABLE_DEBUG
# ifdef  ENABLE_DEBUG
#  define Assert(c,msg) if (!(c)) { fprintf(stderr, "[-] <ASSERTION ERROR> at line %d:  %s\n", __LINE__, msg); exit(1); }
// NOTE(liam): force exit program. basically code should never reach this point.
#  define Throw(msg) { fprintf(stderr, "[-] <THROW> at line %d: %s\n", __LINE__, msg); exit(1); }
# else
#  define ASSERT(c,msg)
#  define Throw(msg) { exit(1); }
# endif
// DEBUG END

#endif //BASE_H
