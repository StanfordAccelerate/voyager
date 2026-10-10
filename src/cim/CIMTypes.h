#pragma once

#include "AccelTypes.h"

namespace cim {

// Declare the lifetime of one ordered CIM resident-set sequence
//
// WeightController emits this at a sequence boundary. CIMProcessor consumes
// set_count consecutive sets in [replay][set] order and releases each set after
// its final one of replay_count traversals
struct WeightDescriptor {
  ac_int<16, false> set_count;
  ac_int<16, false> replay_count;

  static const unsigned int width = 32;

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & set_count;
    m & replay_count;
  }

  inline friend void sc_trace(sc_trace_file* tf,
                              const WeightDescriptor& descriptor,
                              const std::string& name) {
    sc_trace(tf, descriptor.set_count, name + ".set_count");
    sc_trace(tf, descriptor.replay_count, name + ".replay_count");
  }

  inline friend std::ostream& operator<<(std::ostream& os,
                                         const WeightDescriptor& descriptor) {
    os << descriptor.set_count << " " << descriptor.replay_count;
    return os;
  }

  inline friend bool operator==(const WeightDescriptor& lhs,
                                const WeightDescriptor& rhs) {
    return lhs.set_count == rhs.set_count &&
           lhs.replay_count == rhs.replay_count;
  }
};

}  // namespace cim
