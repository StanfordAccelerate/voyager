#define POSIT

#ifdef POSIT
#define INPUT_DATATYPE Posit<8, 1, 8, 16>
#define WEIGHT_DATATYPE Posit<8, 1, 8, 16>
#define ACCUM_DATATYPE Posit<16, 1, 8, 16>
#define INTERMEDIATE_DATATYPE PositFP<8, 16>
#define OUTPUT_DATATYPE Posit<8, 1, 8, 16>
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
