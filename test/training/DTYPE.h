#pragma once

#define FP32

#ifdef FP32
#define DTYPE float
#else
#define DTYPE Posit<8, 1>
#endif