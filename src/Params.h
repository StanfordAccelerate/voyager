#pragma once

struct Params {
  int INPUT_OFFSET;
  int WEIGHT_OFFSET;
  int OUTPUT_OFFSET;
  bool SOFTMAX;

  int SCALE;

  bool TRANSPOSE;

  int VECTOR_OFFSET;
  bool VEC_OP;
  bool VEC_SUB;
  bool VEC_SQUARE;
  bool VEC_REDUCE;
  bool CONST_SCALE;

  int VEC_SCALE_OFFSET;
  int VEC_SUB_OFFSET;
  bool RELU;

  int loops[2][6];
  int inputXLoopIndex[2];
  int inputYLoopIndex[2];
  int reductionLoopIndex[2];
  int weightLoopIndex[2];
  int fxIndex;
  int fyIndex;
  int weightReuseIndex[2];
  bool matMul;
  int STRIDE;
  bool REPLICATION;
  bool MAXPOOL;
  bool BIAS;
  int BIAS_OFFSET;
  bool RESIDUAL;
  int RESIDUAL_OFFSET;
  bool AVGPOOL;

  static const unsigned int width = 12 * 32 + 12 * 32 + 10 * 32 + 14 * 1;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m& INPUT_OFFSET;
    m& WEIGHT_OFFSET;
    m& OUTPUT_OFFSET;
    m& SOFTMAX;
    m& SCALE;
    m& TRANSPOSE;
    m& VECTOR_OFFSET;
    m& VEC_OP;
    m& VEC_SUB;
    m& VEC_SQUARE;
    m& VEC_REDUCE;
    m& CONST_SCALE;
    m& VEC_SCALE_OFFSET;
    m& VEC_SUB_OFFSET;
    m& RELU;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 6; j++) {
        m& loops[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      m& inputXLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& inputYLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& reductionLoopIndex[i];
    }
    for (int i = 0; i < 2; i++) {
      m& weightLoopIndex[i];
    }
    m& fxIndex;
    m& fyIndex;
    for (int i = 0; i < 2; i++) {
      m& weightReuseIndex[i];
    }
    m& matMul;
    m& STRIDE;
    m& REPLICATION;
    m& MAXPOOL;
    m& BIAS;
    m& BIAS_OFFSET;
    m& RESIDUAL;
    m& RESIDUAL_OFFSET;
    m& AVGPOOL;
  }

  inline friend void sc_trace(sc_trace_file* tf, const Params& params,
                              const std::string& name) {
    // TODO
  }

  inline friend std::ostream& operator<<(ostream& os, const Params& params) {
    // TODO
    return os;
  }
};
