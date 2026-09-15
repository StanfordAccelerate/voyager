#include "test/common/GraphUtils.h"

#include <algorithm>
#include <stdexcept>

#include "spdlog/spdlog.h"

// ===========================================================================
// Operations
// ===========================================================================

std::vector<const voyager::PrimOp*> get_prim_ops(const voyager::Operation& op) {
  std::vector<const voyager::PrimOp*> prims;
  if (op.has_prim()) {
    prims.push_back(&op.prim());
  } else if (op.has_fused()) {
    for (const auto& prim : op.fused().op_list()) prims.push_back(&prim);
  }
  return prims;
}

const voyager::PrimOp& get_anchor_op(const voyager::Operation& op) {
  const auto prims = get_prim_ops(op);
  if (prims.empty()) {
    throw std::runtime_error("Accelerator operation " + op.name() +
                             " has no primitive operations.");
  }

  const voyager::PrimOp* anchor = nullptr;
  for (const auto* prim : prims) {
    if (is_gemm_op(strip_namespace(prim->target()))) return *prim;
    if (prim->op() != "call_function") continue;
    if (anchor == nullptr ||
        strip_namespace(anchor->target()) == "dequantize") {
      anchor = prim;
    }
  }

  // Datapath operations normally contain a call_function. Preserve the prior
  // behavior for a standalone host relayout if one reaches the mapper.
  return anchor == nullptr ? *prims.front() : *anchor;
}

bool is_host_bookkeeping(const voyager::PrimOp& prim) {
  const std::string target = strip_namespace(prim.target());
  if (target == "clone") return true;
  if (target != "add") return false;

  const auto input = prim.kwargs().find("input");
  if (input == prim.kwargs().end() || !input->second.has_tensor_box()) {
    return false;
  }
  const std::string& dtype = input->second.tensor_box().box().dtype();
  return dtype == "int32" || dtype == "int64";
}

bool is_datapath(const voyager::Operation& op) {
  for (const auto* prim : get_prim_ops(op)) {
    if (is_host_bookkeeping(*prim)) continue;
    if (prim->op() == "call_function") return true;
  }
  return false;
}

std::string strip_namespace(const std::string& target) {
  const auto pos = target.rfind("::");
  return pos == std::string::npos ? target : target.substr(pos + 2);
}

// ===========================================================================
// Tensor boxes
// ===========================================================================

bool is_semaphore(const voyager::TensorBox& box) {
  return box.memory().level() == voyager::MEMORY_LEVEL_REGISTER;
}

bool is_constant(const voyager::TensorBox& box) {
  // MEMORY_LEVEL_IMMEDIATE is 0, the proto3 default, so a box with no memory
  // field at all reports that level too. Those are fusion intermediates, not
  // constants -- only a box that actually carries a memory message is one.
  return box.has_memory() &&
         box.memory().level() == voyager::MEMORY_LEVEL_IMMEDIATE;
}

bool is_dram(const voyager::TensorBox& box) {
  return box.memory().level() == voyager::MEMORY_LEVEL_DRAM;
}

int partition_of(const voyager::TensorBox& box) {
  switch (box.memory().level()) {
    case voyager::MEMORY_LEVEL_DRAM:
      return DRAM_PARTITION;
    case voyager::MEMORY_LEVEL_SCRATCHPAD:
    case voyager::MEMORY_LEVEL_LOCAL_BUFFER:
      return SRAM_PARTITION;
    default:
      throw std::runtime_error("TensorBox " + box.node() +
                               " is not in an addressable memory");
  }
}

uint32_t banks_of(const voyager::TensorBox& box) {
  return box.bank_count() > 0 ? box.bank_count() : 1;
}

uint64_t bank_stride_of(const voyager::TensorBox& box) {
  return box.bank_stride_bytes();
}

Tensor to_tensor(const voyager::TensorBox& box, int64_t bank) {
  Tensor tensor;
  tensor.node = box.node();
  tensor.shape.assign(box.shape().begin(), box.shape().end());
  tensor.dtype = box.dtype();
  tensor.partition = is_semaphore(box) ? DRAM_PARTITION : partition_of(box);
  tensor.address = box.memory().address() + bank * bank_stride_of(box);
  tensor.materialized = true;
  tensor.is_constant = is_constant(box);
  return tensor;
}

// ===========================================================================
// Operand resolution and data movement
// ===========================================================================

namespace {

// A whole-bank TensorBoxRef selects one bank of a software-pipelined
// allocation, and nothing else. A banked box's referenced dimensions are
// [bank_count, *box.shape]: dimension 0 picks the bank (possibly at runtime)
// and every other dimension spans the whole extent.
//
// That is what makes an operand expressible as a plain base address plus a
// contiguous shape, which is all the kernels and the accelerator's params
// structs can represent. A genuine strided sub-window would need a stride the
// hardware ABI has nowhere to put, so refuse it loudly rather than silently
// reading the wrong bytes.
int64_t select_bank(const voyager::TensorBoxRef& ref, const ScalarEnv& env,
                    const std::string& op_name) {
  if (ref.offsets_size() == 0) return 0;

  const auto reject = [&](const std::string& why) {
    throw std::runtime_error(
        "TensorBoxRef " + ref.box().node() + " in operation " + op_name +
        " is not a whole-bank reference (" + why +
        "). A strided sub-window cannot be expressed as a base address plus a "
        "contiguous shape, so it cannot be lowered. Handle it explicitly in "
        "GraphUtils.cc rather than mis-addressing it.");
  };

  const int bank_dims = banks_of(ref.box()) > 1 ? 1 : 0;
  const int rank = ref.box().shape_size() + bank_dims;
  if (ref.offsets_size() != rank || ref.sizes_size() != rank ||
      ref.strides_size() != rank) {
    reject("rank is not bank_count + box rank");
  }
  if (bank_dims == 1 && ref.sizes(0) != 1) {
    reject("selects more than one bank");
  }

  for (int d = 0; d < rank; d++) {
    if (ref.strides(d) != 1)
      reject("stride != 1 on dimension " + std::to_string(d));
  }
  for (int d = bank_dims; d < rank; d++) {
    const auto& offset = ref.offsets(d);
    if (offset.value_case() != voyager::ScalarValue::kIntValue ||
        offset.int_value() != 0) {
      reject("nonzero offset on dimension " + std::to_string(d));
    }
    if (ref.sizes(d) != ref.box().shape(d - bank_dims)) {
      reject("partial extent on dimension " + std::to_string(d));
    }
  }
  if (bank_dims == 0) return 0;

  const int64_t bank = eval_int(ref.offsets(0), env);
  const uint32_t banks = banks_of(ref.box());
  if (bank < 0 || bank >= static_cast<int64_t>(banks)) {
    throw std::runtime_error("Bank index " + std::to_string(bank) +
                             " out of range for " + ref.box().node() + " (" +
                             std::to_string(banks) + " banks) in " + op_name);
  }
  return bank;
}

// The operand's own shape, when the ref carries one distinct from the box's.
void take_output_shape(Tensor& tensor, const voyager::TensorBoxRef& ref) {
  if (ref.output_shape_size() > 0) {
    tensor.shape.assign(ref.output_shape().begin(), ref.output_shape().end());
  }
}

// The whole-bank convention select_bank enforces, as a non-throwing
// predicate: anything else resolves as a window.
bool is_whole_bank_ref(const voyager::TensorBoxRef& ref) {
  if (ref.offsets_size() == 0) return true;

  const int bank_dims = banks_of(ref.box()) > 1 ? 1 : 0;
  const int rank = ref.box().shape_size() + bank_dims;
  if (ref.offsets_size() != rank || ref.sizes_size() != rank ||
      ref.strides_size() != rank) {
    return false;
  }
  if (bank_dims == 1 && ref.sizes(0) != 1) return false;
  for (int d = 0; d < rank; d++) {
    if (ref.strides(d) != 1) return false;
  }
  for (int d = bank_dims; d < rank; d++) {
    const auto& offset = ref.offsets(d);
    if (offset.value_case() != voyager::ScalarValue::kIntValue ||
        offset.int_value() != 0) {
      return false;
    }
    if (ref.sizes(d) != ref.box().shape(d - bank_dims)) return false;
  }
  return true;
}

int64_t product(const std::vector<int64_t>& values) {
  int64_t total = 1;
  for (const auto& value : values) total *= value;
  return total;
}

std::vector<int64_t> row_major_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int d = static_cast<int>(shape.size()) - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * shape[d + 1];
  }
  return strides;
}

// A windowed reference: a voyager.subview the compiler folded into the
// operand. Its offsets/sizes/strides window the source's dims carried in
// box.shape -- [slot, *dims] for a pipelined buffer -- and output_shape is
// the operand's own shape. Only a window expressible as a base address plus
// a contiguous run resolves; a genuinely strided sub-window still rejects
// loudly.
Tensor resolve_window(const voyager::TensorBoxRef& ref, const ScalarEnv& env,
                      const std::string& op_name) {
  const voyager::TensorBox& box = ref.box();

  const auto reject = [&](const std::string& why) {
    throw std::runtime_error(
        "TensorBoxRef " + box.node() + " in operation " + op_name +
        " is not a resolvable window (" + why +
        "). A strided sub-window cannot be expressed as a base address plus "
        "a contiguous shape, so it cannot be lowered.");
  };

  const int rank = ref.offsets_size();
  if (ref.sizes_size() != rank || ref.strides_size() != rank) {
    reject("offsets, sizes, and strides disagree on rank");
  }
  for (int d = 0; d < rank; d++) {
    if (ref.strides(d) != 1) {
      reject("stride != 1 on dimension " + std::to_string(d));
    }
  }

  // A pipelined buffer's window leads with the slot dim; the bank stride
  // covers it, so only the remaining dims address elements.
  const uint32_t banks = banks_of(box);
  const int bank_dims = banks > 1 ? 1 : 0;
  int64_t bank = 0;
  if (bank_dims == 1) {
    if (ref.sizes(0) != 1) reject("selects more than one bank");
    bank = eval_int(ref.offsets(0), env);
    if (bank < 0 || bank >= static_cast<int64_t>(banks)) {
      reject("bank index " + std::to_string(bank) + " out of range");
    }
  }

  const int dims_rank = rank - bank_dims;
  if (box.shape_size() != dims_rank) {
    reject("window rank does not match the box's dims");
  }
  const std::vector<int64_t> dims(box.shape().begin(), box.shape().end());

  const std::vector<int64_t> strides = row_major_strides(dims);
  int64_t element_offset = 0;
  bool outer_seen = false;
  bool pitched = false;
  for (int d = 0; d < dims_rank; d++) {
    const int64_t offset = eval_int(ref.offsets(bank_dims + d), env);
    const int64_t size = ref.sizes(bank_dims + d);
    if (offset < 0 || offset + size > dims[d]) {
      reject("window exceeds the box on dimension " + std::to_string(d));
    }
    element_offset += offset * strides[d];

    // Base + contiguous shape can express the window only if every dim
    // inside the outermost non-unit one spans its whole extent. A partial
    // innermost run alone is a pitched window: its rows sit a whole source
    // row apart, which Tensor::window_pitch carries.
    if (outer_seen && size != dims[d]) {
      if (d != dims_rank - 1) reject("strided sub-window");
      pitched = true;
    }
    if (size > 1) outer_seen = true;
  }

  Tensor tensor = to_tensor(box, bank);
  take_output_shape(tensor, ref);
  const size_t width = get_width(tensor);
  if ((element_offset * width) % 8 != 0) {
    reject("window does not start on a byte boundary");
  }
  tensor.address += element_offset * width / 8;

  if (pitched) {
    const int64_t run = ref.sizes(rank - 1);
    if (tensor.shape.empty() || tensor.shape.back() != run) {
      reject("the view's shape does not end with the window's run");
    }
    tensor.window_pitch = dims[dims_rank - 1];
    tensor.window_col = element_offset % dims[dims_rank - 1];
  }
  return tensor;
}

std::vector<int64_t> arg_int64s(const voyager::PrimOp& prim,
                                const std::string& key, const ScalarEnv& env) {
  const auto found = prim.kwargs().find(key);
  if (found == prim.kwargs().end()) return {};
  return eval_int_list(found->second.scalar_list(), env);
}

// Moves `count` elements between two buffers, using whole-byte copies when the
// packing lets us and falling back to per-element access when it does not
// (int4/int6 runs can start or end mid-byte).
void copy_run(MemoryInterface* memory, const Tensor& src, int64_t src_index,
              const Tensor& dst, int64_t dst_index, int64_t count) {
  if (count <= 0) return;

  const size_t width = get_width(src);
  const bool byte_aligned = (src_index * width) % 8 == 0 &&
                            (dst_index * width) % 8 == 0 &&
                            (count * width) % 8 == 0;

  if (byte_aligned && src.dtype == dst.dtype) {
    const size_t num_bytes = count * width / 8;
    std::vector<char> buffer(num_bytes);
    memory->read_bytes_from_memory(src.address + src_index * width / 8,
                                   src.partition, num_bytes, buffer.data());
    memory->write_bytes_to_memory(dst.address + dst_index * width / 8,
                                  dst.partition, num_bytes, buffer.data());
    return;
  }

  for (int64_t i = 0; i < count; i++) {
    const float value = memory->read_value(src.partition, src.address,
                                           src_index + i, src.dtype);
    memory->write_value(dst.partition, dst.address, dst_index + i, dst.dtype,
                        value);
  }
}

void fill_run(MemoryInterface* memory, const Tensor& dst, int64_t dst_index,
              int64_t count, float value) {
  for (int64_t i = 0; i < count; i++) {
    memory->write_value(dst.partition, dst.address, dst_index + i, dst.dtype,
                        value);
  }
}

void run_transposed_copy(MemoryInterface* memory, const Tensor& buffer,
                         const Tensor& tile,
                         const std::vector<int64_t>& buffer_shape,
                         const std::vector<int64_t>& buffer_strides,
                         const std::vector<int64_t>& sizes,
                         const std::vector<int64_t>& start, bool is_load,
                         bool has_pad_value, const std::string& name) {
  const int rank = static_cast<int>(sizes.size());
  if (rank < 2) {
    throw std::runtime_error("async_copy " + name +
                             " is transposed but has rank < 2.");
  }
  if (has_pad_value) {
    throw std::runtime_error("async_copy " + name +
                             " is a padded transpose, which is not supported.");
  }
  const int d0 = rank - 2, d1 = rank - 1;

  std::vector<int64_t> tile_shape(sizes.begin(), sizes.end());
  std::swap(tile_shape[d0], tile_shape[d1]);
  if (std::vector<int64_t>(tile.shape.begin(), tile.shape.end()) !=
      tile_shape) {
    throw std::runtime_error("async_copy " + name +
                             " has a transposed tile shape that disagrees with "
                             "the copy sizes.");
  }
  const std::vector<int64_t> tile_strides = row_major_strides(tile_shape);

  const int64_t count = product(sizes);
  std::vector<int64_t> index(rank, 0);
  for (int64_t n = 0; n < count; n++) {
    bool in_bounds = true;
    int64_t buffer_off = 0;
    for (int d = 0; d < rank; d++) {
      const int64_t coord = start[d] + index[d];
      if (coord < 0 || coord >= buffer_shape[d]) {
        in_bounds = false;
        break;
      }
      buffer_off += coord * buffer_strides[d];
    }

    if (!in_bounds) {
      throw std::runtime_error("async_copy " + name + " reads outside " +
                               buffer.node + " without padding.");
    }

    int64_t tile_off = 0;
    for (int d = 0; d < rank; d++) {
      const int64_t coord =
          d == d0 ? index[d1] : (d == d1 ? index[d0] : index[d]);
      tile_off += coord * tile_strides[d];
    }
    if (is_load) {
      copy_run(memory, buffer, buffer_off, tile, tile_off, 1);
    } else {
      copy_run(memory, tile, tile_off, buffer, buffer_off, 1);
    }

    for (int d = rank - 1; d >= 0; d--) {
      if (++index[d] < sizes[d]) break;
      index[d] = 0;
    }
  }
}
}  // namespace

Tensor resolve(const voyager::TensorBoxRef& ref, const ScalarEnv& env,
               const std::string& op_name) {
  const voyager::TensorBox& box = ref.box();

  // Two kinds of operand own no storage in ArrayMemory: a value an earlier
  // PrimOp of this fusion produced, which never leaves the datapath, and a
  // constant, which the kernels read out of tensor_files by node name. Both
  // describe themselves.
  if (!box.has_memory() || is_constant(box)) {
    Tensor tensor;
    tensor.node = box.node();
    tensor.shape.assign(box.shape().begin(), box.shape().end());
    take_output_shape(tensor, ref);
    tensor.dtype = box.dtype();
    tensor.materialized = false;
    tensor.is_constant = is_constant(box);
    return tensor;
  }

  if (!is_whole_bank_ref(ref)) return resolve_window(ref, env, op_name);

  const int64_t bank = select_bank(ref, env, op_name);
  // The copy's sizes and indices are counted in the operand's own dims.
  Tensor tensor = to_tensor(box, bank);
  take_output_shape(tensor, ref);
  return tensor;
}

Tensor resolve(const voyager::PrimOp& op, const std::string& key,
               const ScalarEnv& env) {
  return resolve(arg(op, key).tensor_box(), env, op.name());
}

std::vector<Tensor> resolve_outputs(const voyager::Operation& op,
                                    const ScalarEnv& env) {
  std::vector<Tensor> outputs;
  for (const auto& output : op.outputs()) {
    if (output.has_destination()) {
      outputs.push_back(resolve(output.destination(), env, op.name()));
    } else if (output.has_tensor_box()) {
      outputs.push_back(to_tensor(output.tensor_box()));
    } else {
      throw std::runtime_error("Accelerator operation " + op.name() +
                               " yields a scalar, which it cannot.");
    }
  }
  if (outputs.empty()) {
    throw std::runtime_error("Accelerator operation " + op.name() +
                             " writes nothing.");
  }
  return outputs;
}

int64_t semaphore_slots(const voyager::TensorBox& box) {
  int64_t slots = banks_of(box);
  for (const auto dim : box.shape()) slots *= dim;
  return slots;
}

int64_t resolve_bank(const voyager::TensorBoxRef& ref, const ScalarEnv& env,
                     const std::string& op_name) {
  const voyager::TensorBox& box = ref.box();
  const int dims = box.shape_size();
  if (dims == 0 || ref.offsets_size() == 0)
    return select_bank(ref, env, op_name);

  const auto reject = [&](const std::string& why) {
    throw std::runtime_error("Semaphore TensorBoxRef " + box.node() +
                             " in operation " + op_name + " (" + why + ").");
  };

  // [bank, *dims], one counter selected.
  const int bank_dims = banks_of(box) > 1 ? 1 : 0;
  if (ref.offsets_size() != dims + bank_dims ||
      ref.sizes_size() != dims + bank_dims) {
    reject("rank is not bank_count + the semaphore array's rank");
  }
  for (int d = 0; d < ref.sizes_size(); d++) {
    if (ref.sizes(d) != 1) reject("selects more than one counter");
  }

  int64_t bank = 0;
  if (bank_dims == 1) {
    bank = eval_int(ref.offsets(0), env);
    const uint32_t banks = banks_of(box);
    if (bank < 0 || bank >= static_cast<int64_t>(banks)) {
      reject("bank index " + std::to_string(bank) + " is out of range for " +
             std::to_string(banks) + " banks");
    }
  }

  int64_t index = 0;
  for (int d = 0; d < dims; d++) {
    const int64_t extent = box.shape(d);
    const int64_t offset = eval_int(ref.offsets(d + bank_dims), env);
    if (offset < 0 || offset >= extent) {
      reject("index " + std::to_string(offset) +
             " is out of range on "
             "dimension " +
             std::to_string(d) + " (extent " + std::to_string(extent) + ")");
    }
    index = index * extent + offset;
  }

  int64_t elements = 1;
  for (const auto dim : box.shape()) elements *= dim;
  return bank * elements + index;
}

// ===========================================================================
// Data movement
// ===========================================================================

void run_async_copy(const voyager::PrimOp& prim, const ScalarEnv& env,
                    MemoryInterface* memory) {
  const std::string& name = prim.name();

  const Tensor src = resolve(prim, "src", env);
  const Tensor dst = resolve(prim, "dst", env);

  std::vector<int64_t> sizes = arg_int64s(prim, "sizes", env);
  std::vector<int64_t> strides = arg_int64s(prim, "strides", env);
  if (strides.empty()) strides = sizes;

  // An absent `dims` means every dimension is indexed. An empty-but-present
  // `dims` means none is -- a distinction proto3 only makes through has_*.
  const bool has_dims = prim.kwargs().count("dims") > 0;
  const std::vector<int64_t> dims = arg_int64s(prim, "dims", env);
  const std::vector<int64_t> indices = arg_int64s(prim, "indices", env);

  // The data-dependent extent of a sparse copy: the tile is `sizes` big, but
  // only its leading `count` region is valid (a CSR row's nnz), so the copy
  // moves that region and leaves the rest of the tile untouched. Matches the
  // compiler's async_copy, where `count` slices both the tile and the block:
  // dst[block][payload] = src[payload]. Absent means the whole tile; a
  // transposed copy ignores it, exactly as the compiler does.
  const std::vector<int64_t> count = arg_int64s(prim, "count", env);

  const bool transposed =
      prim.kwargs().count("transposed") > 0 &&
      to_bool(eval(prim.kwargs().at("transposed").scalar(), env));

  const bool has_pad_value = prim.kwargs().count("pad_value") > 0;
  const float pad_value =
      has_pad_value ? static_cast<float>(to_double(
                          eval(prim.kwargs().at("pad_value").scalar(), env)))
                    : 0.0f;

  // Match the compiler's async_copy contract: a transposed source is always
  // the buffer; otherwise a source whose shape equals `sizes` is the tile of
  // a store.
  const int64_t tile_elements = product(sizes);
  const std::vector<int64_t> src_shape(src.shape.begin(), src.shape.end());
  const bool src_is_buffer = transposed || src_shape != sizes;

  const Tensor& buffer = src_is_buffer ? src : dst;
  const Tensor& tile = src_is_buffer ? dst : src;

  if (get_size(tile) != tile_elements) {
    throw std::runtime_error(
        "async_copy " + name + ": tile operand " + tile.node + " holds " +
        std::to_string(get_size(tile)) + " elements but the copy moves " +
        std::to_string(tile_elements) + ".");
  }

  // The operand carries the shape the copy counts its tiles in.
  const std::vector<int64_t> buffer_shape(buffer.shape.begin(),
                                          buffer.shape.end());
  const int rank = static_cast<int>(sizes.size());
  if (static_cast<int>(buffer_shape.size()) != rank) {
    throw std::runtime_error(
        "async_copy " + name + ": buffer operand " + buffer.node +
        " has rank " + std::to_string(buffer_shape.size()) +
        " but the copy moves a rank-" + std::to_string(rank) + " tile.");
  }

  // Scatter the tile index across the dimensions the copy actually tiles.
  std::vector<int64_t> block(rank, 0);
  if (!has_dims) {
    if (static_cast<int>(indices.size()) != rank) {
      throw std::runtime_error("async_copy " + name +
                               ": indices rank does not match sizes rank.");
    }
    block = indices;
  } else {
    for (size_t i = 0; i < dims.size(); i++) block[dims[i]] = indices[i];
  }

  const std::vector<int64_t> pad = arg_int64s(prim, "pad", env);
  std::vector<int64_t> start(rank);
  for (int d = 0; d < rank; d++) {
    start[d] = block[d] * strides[d] - (pad.empty() ? 0 : pad[d]);
  }

  const bool is_load = src_is_buffer;
  const std::vector<int64_t> buffer_strides = row_major_strides(buffer_shape);

  // A transposed copy reorders the tile's last two axes, so it cannot use the
  // contiguous inner-run path below; hand it to the element-wise transposer.
  if (transposed) {
    run_transposed_copy(memory, buffer, tile, buffer_shape, buffer_strides,
                        sizes, start, is_load, has_pad_value, name);
    return;
  }

  if (!count.empty() && static_cast<int>(count.size()) != rank) {
    throw std::runtime_error("async_copy " + name +
                             ": count rank does not match sizes rank.");
  }
  const std::vector<int64_t>& extent = count.empty() ? sizes : count;

  const int64_t inner = extent[rank - 1];
  int64_t outer = 1;
  for (int d = 0; d + 1 < rank; d++) outer *= extent[d];

  std::vector<int64_t> index(rank, 0);
  for (int64_t o = 0; o < outer; o++) {
    // Where this run starts in the buffer, and whether the outer dimensions
    // land inside it at all: a padded halo can hang off any edge.
    bool in_bounds = true;
    int64_t buffer_base = 0;
    for (int d = 0; d + 1 < rank; d++) {
      const int64_t coord = start[d] + index[d];
      if (coord < 0 || coord >= buffer_shape[d]) {
        in_bounds = false;
        break;
      }
      buffer_base += coord * buffer_strides[d];
    }

    // Clip the run itself against the innermost extent.
    const int64_t base = start[rank - 1];
    int64_t lo = 0;
    int64_t hi = inner;
    if (in_bounds) {
      lo = std::max<int64_t>(0, -base);
      hi = std::min<int64_t>(inner, buffer_shape[rank - 1] - base);
      hi = std::max(lo, hi);
    } else {
      lo = hi = 0;
    }

    // The tile keeps its full `sizes` layout even when `count` trims the
    // copy, so its offsets walk `sizes`, not the copied extent.
    int64_t tile_base = 0;
    for (int d = 0; d < rank; d++) tile_base = tile_base * sizes[d] + index[d];

    if (is_load) {
      if (lo > 0 || hi < inner) {
        if (!has_pad_value) {
          throw std::runtime_error("async_copy " + name + " reads outside " +
                                   buffer.node + " but carries no pad_value.");
        }
        fill_run(memory, tile, tile_base, lo, pad_value);
        fill_run(memory, tile, tile_base + hi, inner - hi, pad_value);
      }
      copy_run(memory, buffer, buffer_base + base + lo, tile, tile_base + lo,
               hi - lo);
    } else {
      if (lo != 0 || hi != inner) {
        throw std::runtime_error("async_copy " + name + " stores outside " +
                                 buffer.node + ": run [" +
                                 std::to_string(base) + ", " +
                                 std::to_string(base + inner) + ") of extent " +
                                 std::to_string(buffer_shape[rank - 1]) +
                                 " (clipped to [" + std::to_string(base + lo) +
                                 ", " + std::to_string(base + hi) + ")).");
      }
      copy_run(memory, tile, tile_base, buffer, buffer_base + base, inner);
    }

    for (int d = rank - 2; d >= 0; d--) {
      if (++index[d] < extent[d]) break;
      index[d] = 0;
    }
  }
}

void zero_buffer(const Tensor& tensor, uint32_t banks, uint64_t bank_stride,
                 MemoryInterface* memory) {
  const uint64_t span = banks > 1 ? banks * bank_stride : get_num_bytes(tensor);
  std::vector<char> zeros(span, 0);
  memory->write_bytes_to_memory(tensor.address, tensor.partition, span,
                                zeros.data());
}
