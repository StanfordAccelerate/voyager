// Shared CIM type definitions for macro modes and implementation selection

`ifndef CIM_TYPEDEFS_SVH
`define CIM_TYPEDEFS_SVH

typedef enum logic {
  CIM_MODE_BIT_PARALLEL = 1'b0,
  CIM_MODE_BIT_SERIAL   = 1'b1
} cim_mode_t;

typedef enum logic {
  CIM_MACRO_WRAPPER_IMPL_MODEL           = 1'b0,
  CIM_MACRO_WRAPPER_IMPL_CIM_MACRO_1     = 1'b1
} cim_macro_wrapper_impl_t;

`endif  // CIM_TYPEDEFS_SVH
