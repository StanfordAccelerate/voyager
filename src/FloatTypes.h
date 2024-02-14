// This file contains additional Float functions which are not natively supported by <ac_float.h>
// These additional functions use existing <ac_float.h> primitives as building blocks

#pragma once

#include <ac_float.h>
#include <ac_int.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_pow_pwl.h>
// #include <connections/marshaller.h>

// For Floating point number there is always 1 implicit hidden bit 
#define hidden_bits 1

template <int mantissa, int exp>
//  : public ac_float<mantissa+hidden_bits, hidden_bits, exp, AC_RND>
class Float{
    public:

    // ac_float<W,I,E,Q>- mantissa: ac_fixed<W,I,true> exp: ac_int<E,true> quant : default trunc 
    typedef ac_float<mantissa+hidden_bits, hidden_bits, exp, AC_RND> ac_float_rep; 

    // TODO Set Float to Fixed type
    typedef ac_fixed<2*mantissa, mantissa, true> ac_float_to_fixed_rep;

    // Set Accumulation datatype size for Floats to floating point type
    typedef Float<mantissa, exp> AccumulationDatatype;

    Float();
    Float(ac_float_rep input_float_rep);
    template<int i_mantissa, int i_exp>
    Float(Float<i_mantissa, i_exp> rhs);

    // ac_int<mantissa+exp+hidden_bits,true> bits(){
    //     return float_val.data_ac_int();
    // }

    void negate(){
        float_val = -float_val;
    }

    void relu(){
        if (float_val < 0) float_val = 0;
    }

    void masked_relu(const Float &mask){
        if (mask == 0) float_val = 0;
    }

    void setbits(int i) { float_val = i; }


    // void setbits(Float<mantissa, exp> i) { float_val = i; }

    void setZero(){
        float_val = 0;
    }

    void custom_converted_reciprocal(){
        float_val = 1/float_val;
    }    

    void reciprocal(){
        float_val = 1/float_val;
    }

    void exponent(){
        // convert to fixed point
        ac_float_to_fixed_rep converted_to_fixed = float_val.to_ac_fixed();
        ac_float_to_fixed_rep exponent_in_fixed;
        // take fixed point exponent
        ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
        // convert back to float
        float_val = exponent_in_fixed.to_ac_float();
    }   

    // log_mult not implemented for float type

    Float inv_sqrt(){
        ac_float_rep float_val_inv_sqrt; 
        ac_math::ac_inverse_sqrt_pwl(float_val, float_val_inv_sqrt);
        return float_val_inv_sqrt;
    }

    void sigmoid(){
        ac_float_to_fixed_rep converted_to_fixed = float_val.to_ac_fixed();
        ac_float_to_fixed_rep sigmoid_in_fixed;        
        ac_math::ac_sigmoid_pwl(converted_to_fixed, sigmoid_in_fixed);
        float_val = sigmoid_in_fixed.to_ac_float();
    }

  Float operator+(const Float &rhs);
  Float operator*(const Float &rhs);
  Float operator/(const Float &rhs);

  Float &operator+=(const Float &rhs);
  Float &operator-=(const Float &rhs);
  Float &operator*=(const Float &rhs);
  Float &operator/=(const Float &rhs);

  bool operator<(const Float &rhs) const;

  operator float() const;

// SystemC is not compatible with C++17
#ifndef NO_SYSC
    template <unsigned int Size>
    void Marshall(Marshaller<Size> &m) {
        m &float_val;
    }
#endif

    // static const unsigned int width = ac_float_rep::width;
    static constexpr int width = mantissa + exp + hidden_bits;
    private:
    ac_float_rep float_val;


};

template <int mantissa, int exp>
inline std::ostream &operator<<(std::ostream &os, const Float<mantissa, exp> &val) {
#ifndef __SYNTHESIS__
  os << static_cast<float>(val.float_val) << " ";
#else
  os << val.bits << " ";
#endif
  return os;
}

template <int mantissa, int exp>
Float<mantissa, exp>::Float(ac_float_rep input_float_rep){
    float_val = input_float_rep;
}

template <int mantissa, int exp>
Float<mantissa, exp>::Float(){
    float_val = 0;
}


template<int mantissa, int exp>
template<int i_mantissa, int i_exp>
Float<mantissa, exp>::Float(Float<i_mantissa, i_exp> rhs){
    float_val = rhs;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> Float<mantissa, exp>::operator+(
    const Float<mantissa, exp> &rhs) {
  return *this + rhs;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> Float<mantissa, exp>::operator*(
    const Float<mantissa, exp> &rhs) {
  return *this * rhs;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> Float<mantissa, exp>::operator/(
    const Float<mantissa, exp> &rhs) {
  return *this / rhs;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator+=(
    const Float<mantissa, exp> &rhs) {
  return *this + rhs;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator-=(
    const Float<mantissa, exp> &rhs) {
  return *this - rhs;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator*=(
    const Float<mantissa, exp> &rhs) {
  return *this * rhs;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator/=(
    const Float<mantissa, exp> &rhs) {
  return *this / rhs;
}

template <int mantissa, int exp>
inline bool Float<mantissa, exp>::operator<(const Float<mantissa, exp> &rhs) const {
  return (*this < rhs);
}
