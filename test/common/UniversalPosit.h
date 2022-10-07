#pragma once

#ifndef NO_UNIVERSAL
#include <universal/number/posit/posit.hpp>
typedef sw::universal::posit<8, 1> UniversalPosit;
typedef sw::universal::posit<16, 1> UniversalPositAccum;

#else
typedef float UniversalPosit;
typedef float UniversalPositAccum;

#endif