#ifndef DEFINES_H
#define DEFINES_H

#include <stdio.h>
#include <stdint.h>

// NOTE(liam): context cracking
// used for context-specific settings
# if defined(_WIN32) || defined(_WIN64)
#  define OS_WINDOWS 1
# elif defined(__linux__) || defined(__gnu_linux__) || defined(__unix__)
#  define OS_LINUX 1
# elif defined(__APPLE__) && defined(__MACH__)
# error "Unsupported Operating System."
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

# if defined(__clang__)
#  define COMPILER_CLANG 1
# elif defined(__GNUC__)
#  define COMPILER_GCC 1
# elif defined(__TINYC__)
#  define COMPILER_TCC 1
# else
/*#  error "Unsupported Compiler."*/
# endif

# if defined(OS_WINDOWS)
#  define CLEAR system("cls")
# elif defined(OS_LINUX) || defined(OS_MAC)
#  define CLEAR system("clear")
#  define EXTERN_C extern "C"
# else
#  define CLEAR 0
# endif

# if defined(COMPILER_CLANG)
#  define FILE_NAME __FILE_NAME__
#  define threadvar __thread
# else
#  define FILE_NAME __FILE__
# endif

// aliases
# define global     static
# define local      static

# define EXTERN_C_START extern "C" {
# define EXTERN_C_END  }

// some funny redefs
# define AND &&
# define OR ||
# define NOT !
# define elif else if
# define is =

# define true 1
# define false 0

# define null 0

typedef uint8_t      uint8;  // unsigned char
typedef uint16_t     uint16; // unsigned int
typedef uint32_t     uint32; // unsigned long int
typedef uint64_t     uint64; // unsigned long long int

typedef int8_t       int8;   // signed char
typedef int16_t      int16;  // signed int
typedef int32_t      int32;  // signed long int
typedef int64_t      int64;  // signed long long int

typedef uint8  u8;
typedef uint16 u16;
typedef uint32 u32;
typedef uint64 u64;

typedef int8   i8;
typedef int16  i16;
typedef int32  i32;
typedef int64  i64;

typedef float       float32;
typedef double      float64;
typedef int32       bool32;
typedef size_t      memory_index;

typedef void VoidFunc(void);

// NOTE(liam): EXPERIMENTAL: here for demo purposes
int ap_int(int a, int b);
int ap_float(float a, float b);
# define OVERLOADING
# ifdef  OVERLOADING
#  define ap(a, b) _Generic((a),    \
    int     :   ap_int((a), (b)),   \
    float   :   ap_float((a), (b))  \
)
# endif

// NOTE(liam): general macros
# define Swap(a, b)      (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
# define Min(a, b)       ((a) < (b) ? (a) : (b))
# define Max(a, b)       ((a) > (b) ? (a) : (b))
# define Limit(a, x, b)  ((x) = (x) < (a) ? (a) : (x) > (b) ? (b) : (x))
# define ClampUp(a, b)   Max(a, b)
# define ClampDown(a, b) Min(a, b)
# define Abs(a)          (((a) < 0) ? -(a) : (a))

// NOTE(liam): increment/decrement 'a' per call, limiting when 'a == x'
# define SaturateUp(a, x)   ((a) > (x) ? (a = x) : (a++))
# define SaturateDown(a, x) ((a) < (x) ? (a = x) : (a--))

// TODO(liam): check if this works
# define Implies(a, b)      (!(a) || (b))

# define ArrayCount(a) (sizeof(a)/sizeof(*(a)))
# define IsPowerOfTwo(n) (((n) & ((n) - 1)) == 0)

# define Kilobytes(V) ((V)*1024LL)
# define Megabytes(V) (Kilobytes(V)*1024LL)
# define Gigabytes(V) (Megabytes(V)*1024LL)
# define Terabytes(V) (Gigabytes(V)*1024LL)

# define Statement(S) do{ S }while(0)

# define Stringify_(S) #S
# define Stringify(S) Stringify_(S)
# define Glue_(A,B) A##B
# define Glue(A,B) Glue_(A,B)

# define IntFromPtr(p) (unsigned long long)((char*)p - (char*)0)
# define PtrFromInt(n) (void*)((char*)0 + (n))

// NOTE(liam): behavior of assert.
# if !defined(AssertBreak)
#  include <stdlib.h>
#  define AssertBreak(c) { fprintf(stderr, "%s:%d: failed assertion '%s'\n.", __FILE__, __LINE__, #c); }
# endif

// DEBUG START
# define ENABLE_DEBUG
# ifdef  ENABLE_DEBUG
#define Assert(c) if(!(c)) { AssertBreak(c); }

// NOTE(liam): force exit program. basically code should never reach this point.
#  define Throw(msg) { fprintf(stderr, "[*] <THROW> at line %d: %s\n", __LINE__, #msg); Statement(AssertBreak();); }
# else
#  define Assert(c)
#  define Throw(msg)
# endif
// DEBUG END

#endif //DEFINES_H
