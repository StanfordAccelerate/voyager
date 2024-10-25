#pragma once

#include <mc_connections.h>

/*
 * This file provides a way to conditionally instantiate Connections. It's
 * similar to the standard Connections usage, but takes in an extra boolean
 * template parameter. If the boolean is true, the connection is instantiated as
 * normal. If the boolean is false, an EmptyType is used instead. This allows
 * for easy conditional instantiation of Connections. Inspired by
 * https://brevzin.github.io/c++/2021/11/21/conditional-members/
 */

// EmptyType can take any number of arguments and will ignore them all.
struct EmptyType {
  constexpr EmptyType(auto &&...) {}
};

template <typename Module, bool condition>
using ConditionalModule = std::conditional_t<condition, Module, EmptyType>;

namespace Connections {
template <typename Message, bool condition>
using ConditionalIn =
    std::conditional_t<condition, Connections::In<Message>, EmptyType>;

template <typename Message, bool condition>
using ConditionalOut =
    std::conditional_t<condition, Connections::Out<Message>, EmptyType>;

template <typename Message, bool condition>
using ConditionalCombinational =
    std::conditional_t<condition, Connections::Combinational<Message>,
                       EmptyType>;
};  // namespace Connections