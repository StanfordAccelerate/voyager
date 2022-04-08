#define POSIT

#ifdef POSIT
#define INPUT_DATATYPE Posit<8, 1>
#define WEIGHT_DATATYPE Posit<8, 1>
#define ACCUM_DATATYPE Posit<16, 1>
#define OUTPUT_DATATYPE Posit<8, 1>
#else
#define INPUT_DATATYPE ac_int<8, true>
#define WEIGHT_DATATYPE ac_int<8, true>
#define ACCUM_DATATYPE ac_int<8, true>
#define OUTPUT_DATATYPE ac_int<8, true>
#endif

#define DIMENSION 16
#define INPUT_BUFFER_SIZE 1024
#define WEIGHT_BUFFER_SIZE 1024
#define ACCUMULATION_BUFFER_SIZE 1024

// #define PIPE_INPUT 1
// #define INPUT_SCALING
// #define WEIGHT_SCALING
