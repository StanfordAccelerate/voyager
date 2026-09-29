// Signal-level helpers let consumers inspect requests before accepting
// them. A blocking Connections Pop() accepts a request before exposing it.

#pragma once

#include <mc_connections.h>

#include "TypeToBits.h"

// ConnectionsSignal isolates signal-level operations needed for same-edge admission
namespace ConnectionsSignal {

template <typename T>
using In = Connections::In<T, Connections::SYN_PORT>;

// Decode the presented payload without accepting it or waiting for valid
template <typename T>
T peek(const In<T>& input) {
  return BitsToType<T>(input.dat.read());
}

// Return whether an input currently presents a valid payload
template <typename T>
bool valid(const In<T>& input) {
  return input.vld.read();
}

// Return whether an input transfers on the current edge
template <typename T>
bool fired(const In<T>& input) {
  return input.vld.read() && input.rdy.read();
}

// Drive combinational admission for an input
template <typename T>
void set_ready(In<T>& input, bool ready) {
  input.rdy.write(ready);
}

}  // namespace ConnectionsSignal
