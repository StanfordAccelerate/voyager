// This file contains additional Float functions which are not natively supported by <ac_float.h>
// These additional functions use existing <ac_float.h> primitives as building blocks

#pragma once

#include <ac_float.h>
#include <ac_int.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_pow_pwl.h>
// #include <TypeToBits.h>
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

    // TODO Set Float to Fixed type
    typedef ac_fixed<2*mantissa, mantissa, false> ac_float_to_fixed_rep_out;

    // Set Accumulation datatype size for Floats to floating point type
    typedef Float<mantissa, exp> AccumulationDatatype;

    Float();
    Float(ac_float_rep input_float_rep);
    template<int i_mantissa, int i_exp>
    Float(Float<i_mantissa, i_exp> rhs);

    // ac_int<mantissa+exp+hidden_bits,true> bits_rep(){
    //     return float_val.data_ac_int();
    // }

    ac_int<mantissa + exp + hidden_bits, false> bits_rep() const { 
      ac_int<mantissa + exp + hidden_bits, false> bit_expression;
      // bit_expression = BitsToType<ac_int<mantissa + exp + hidden_bits, false> >
      //                   (TypeToBits<ac_float_rep>(float_val));
      // bit_expression = static_cast<ac_int<mantissa + exp + hidden_bits, false> >
      //                   (static_cast<sc_lv<mantissa + exp + hidden_bits> >(float_val));
      // <exp> <mantissa>
      //  E    W  
      bit_expression.set_slc(0, float_val.mantissa().to_ac_int());
      bit_expression.set_slc(mantissa + hidden_bits, float_val.exp());
      return bit_expression;
    }   

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
        ac_float_to_fixed_rep_out exponent_in_fixed;
        // take fixed point exponent
        ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
        // convert back to float
        float_val = static_cast<ac_float_rep>(exponent_in_fixed);
    }

    void sqrt() {
	throw "Float sqrt function not implemented.";
    }

    // log_mult not implemented for float type

    Float inv_sqrt(){
        ac_float_rep float_val_inv_sqrt; 
        ac_math::ac_inverse_sqrt_pwl(float_val, float_val_inv_sqrt);
        return float_val_inv_sqrt;
    }

    Float max1(){
        ac_float_rep one;
        one = 1.0;
        if(float_val > one){
          float_val = one;
        } 
        return float_val;
    }

    void sigmoid(){
        ac_float_to_fixed_rep converted_to_fixed = float_val.to_ac_fixed();
        ac_float_to_fixed_rep_out sigmoid_in_fixed;        
        ac_math::ac_sigmoid_pwl(converted_to_fixed, sigmoid_in_fixed);
        float_val = static_cast<ac_float_rep>(sigmoid_in_fixed);
    }

    void expScale(ac_int<8, false> offset){
      float_val.set_exp(float_val.exp() + offset);
    }    

  Float operator+(const Float &rhs);
  Float operator*(const Float &rhs);
  Float operator/(const Float &rhs);

  Float &operator+=(const Float &rhs);
  Float &operator-=(const Float &rhs);
  Float &operator*=(const Float &rhs);
  Float &operator/=(const Float &rhs);

  bool operator<(const Float &rhs) const;

  operator float() const{
    return float_val.to_float();
  }

// SystemC is not compatible with C++17
#ifndef NO_SYSC
    template <unsigned int Size>
    void Marshall(Marshaller<Size> &m) {
      // std::cout << "called marshall" << std::endl;
      // m &float_val;
      // float_val.Marshall(m);
      Wrapped<ac_float_rep> wrapped_float_val(float_val);
      m &wrapped_float_val;     
    }
#endif

    // static const unsigned int width = ac_float_rep::width;
    static constexpr int width = mantissa + exp + hidden_bits;
    private:
    ac_float_rep float_val;


};

template <int mantissa, int exp>
inline std::ostream &operator<<(std::ostream &os, const Float<mantissa, exp> &val) {
  os << val.bits_rep() << " ";
  return os;
}

template <int mantissa, int exp>
Float<mantissa, exp> exponent(Float<mantissa, exp> element){
  typedef ac_float<mantissa+hidden_bits, hidden_bits, exp, AC_RND> ac_float_rep;
  typedef ac_fixed<2*mantissa, mantissa, true> ac_float_to_fixed_rep;
  typedef ac_fixed<2*mantissa, mantissa, false> ac_float_to_fixed_out_rep;
  // convert to fixed point
  ac_float_to_fixed_rep converted_to_fixed =
     static_cast<ac_float_rep>(element).to_ac_fixed();
  ac_float_to_fixed_out_rep exponent_in_fixed;
  // take fixed point exponent
  ac_math::ac_exp_pwl(converted_to_fixed, exponent_in_fixed);
  // convert back to float
  return static_cast<ac_float_rep>(exponent_in_fixed);
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
    float_val = static_cast<ac_float_rep>(rhs);
}

template<int mantissa, int exp>
inline Float<mantissa, exp> Float<mantissa, exp>::operator+(
    const Float<mantissa, exp> &rhs) {
  AccumulationDatatype op1 = *this;
  AccumulationDatatype op2 = rhs;
  return op1.float_val.add(op1.float_val, op2.float_val);
}

template<int mantissa, int exp>
inline Float<mantissa, exp> Float<mantissa, exp>::operator*(
    const Float<mantissa, exp> &rhs) {
  AccumulationDatatype op1 = *this;
  AccumulationDatatype op2 = rhs;
  return static_cast<ac_float_rep>(op1.float_val * op2.float_val);
}

template<int mantissa, int exp>
inline Float<mantissa, exp> Float<mantissa, exp>::operator/(
    const Float<mantissa, exp> &rhs) {
  AccumulationDatatype op1 = *this;
  AccumulationDatatype op2 = rhs;
  return op1.float_val / op2.float_val;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator+=(
    const Float<mantissa, exp> &rhs) {
  *this = *this + rhs;
  return *this;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator-=(
    const Float<mantissa, exp> &rhs) { 
  *this = static_cast<Float<mantissa, exp>>(*this - rhs);
  return *this;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator*=(
    const Float<mantissa, exp> &rhs) {
  *this = *this * rhs;
  return *this;
}

template<int mantissa, int exp>
inline Float<mantissa, exp> &Float<mantissa, exp>::operator/=(
    const Float<mantissa, exp> &rhs) {
  *this = *this / rhs;
  return *this;
}

template <int mantissa, int exp>
inline bool Float<mantissa, exp>::operator<(const Float<mantissa, exp> &rhs) const {
  return (*this < rhs);
}

template <int mantissa_i, int exp_i, int mantissa_o, int exp_o>
typename Float<mantissa_o, exp_o>::AccumulationDatatype decomposed_fma(
    const typename Float<mantissa_i, exp_i>::AccumulationDatatype &a,
    const typename Float<mantissa_i, exp_i>::AccumulationDatatype &b,
    const typename Float<mantissa_o, exp_o>::AccumulationDatatype &c) {

  return  static_cast<typename Float<mantissa_o, exp_o>::AccumulationDatatype >(a * b) + c;
  // if (c.isZero()) {
  //   return typename Posit<nbits2, es2>::AccumulationDatatype(product);
  // } else {
  //   PositFP<8, abits + 1> sum = PositFP<8, fbits2>(product) + c;
  //   return typename Posit<nbits2, es2>::AccumulationDatatype(sum);
}
