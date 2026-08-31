#include "test/common/Interpreter.h"

#include <algorithm>
#include <stdexcept>

#include "spdlog/spdlog.h"
#include "test/common/GraphUtils.h"
#include "test/common/Utils.h"

namespace {

// The host-side arithmetic that drives tiling: loop counters, tile indices,
// bank selection, and every branch predicate. All of it runs on the control
// processor, never on the accelerator.
Scalar apply_scalar_op(const std::string& target, const Scalar& a,
                       const Scalar& b) {
  const bool is_float =
      std::holds_alternative<double>(a) || std::holds_alternative<double>(b);

  if (target == "add") {
    if (is_float) return to_double(a) + to_double(b);
    return to_int(a) + to_int(b);
  }
  if (target == "sub") {
    if (is_float) return to_double(a) - to_double(b);
    return to_int(a) - to_int(b);
  }
  if (target == "mul") {
    if (is_float) return to_double(a) * to_double(b);
    return to_int(a) * to_int(b);
  }
  if (target == "mod") {
    const int64_t divisor = to_int(b);
    if (divisor == 0) throw std::runtime_error("Scalar mod by zero");
    // Python/sympy floor semantics, which is what the compiler emitted against.
    const int64_t remainder = to_int(a) % divisor;
    return (remainder != 0 && ((remainder < 0) != (divisor < 0)))
               ? remainder + divisor
               : remainder;
  }
  if (target == "floordiv") {
    const int64_t divisor = to_int(b);
    if (divisor == 0) throw std::runtime_error("Scalar floordiv by zero");
    const int64_t quotient = to_int(a) / divisor;
    const int64_t remainder = to_int(a) % divisor;
    return (remainder != 0 && ((remainder < 0) != (divisor < 0))) ? quotient - 1
                                                                  : quotient;
  }

  if (target == "eq")
    return is_float ? to_double(a) == to_double(b) : to_int(a) == to_int(b);
  if (target == "ne")
    return is_float ? to_double(a) != to_double(b) : to_int(a) != to_int(b);
  if (target == "lt")
    return is_float ? to_double(a) < to_double(b) : to_int(a) < to_int(b);
  if (target == "le")
    return is_float ? to_double(a) <= to_double(b) : to_int(a) <= to_int(b);
  if (target == "gt")
    return is_float ? to_double(a) > to_double(b) : to_int(a) > to_int(b);
  if (target == "ge")
    return is_float ? to_double(a) >= to_double(b) : to_int(a) >= to_int(b);

  if (target == "and_") return to_bool(a) && to_bool(b);
  if (target == "or_") return to_bool(a) || to_bool(b);

  throw std::runtime_error("Unsupported scalar operation: " + target);
}

// Cycles one execution of an accelerator operation would take with no stalls.
//
// The operands are one tile, so this is the cost of one tile. The walk runs the
// operation once per tile, which is where the old num_tiles multiplier went.
long ideal_cycles(const voyager::Operation& operation, const ScalarEnv& env,
                  bool& is_matrix) {
  const auto& first_op = get_anchor_op(operation);
  const auto outputs = resolve_outputs(operation, env);
  const auto& output = outputs.back();
  const std::string& target = strip_namespace(first_op.target());

  is_matrix = is_gemm_op(target);

  if (is_matrix) {
    const bool is_matmul = target.find("matmul") != std::string::npos;
    const auto& weight = resolve(first_op, is_matmul ? "other" : "weight", env);
    const auto& weight_shape = weight.shape;

    // The total number of MACs is X * Y * C * FX * FY * K.
    const long num_macs = get_size(output) * get_size(weight);

    if (is_fc_layer(first_op)) {
      if (weight_shape.size() < 2) {
        throw std::runtime_error(
            "MVM weight must have at least two dimensions.");
      }
      const int K = weight_shape[weight_shape.size() - 2];
      const auto& input = resolve(first_op, "input", env);
      const int width =
          input.dtype == "bfloat16" ? VECTOR_UNIT_WIDTH : MV_UNIT_WIDTH;
      return num_macs / K / width;
    }

    const int K = weight_shape.back();
    return num_macs / K / (IC_DIMENSION * OC_DIMENSION);
  }

  long cycles = get_size(output) / VECTOR_UNIT_WIDTH;

  if (target == "softmax") {
    cycles *= 3;
  } else if (target == "layer_norm") {
    cycles *= 4;
  } else if (target == "calculate_mx_qparam") {
    cycles *= arg_int(first_op, "block_size", env);
  } else if (target == "max_pool2d") {
    for (const auto& k : arg_ints(first_op, "kernel_size", env)) cycles *= k;
  } else if (target == "adaptive_avg_pool2d") {
    const auto& input_shape = resolve(first_op, "input", env).shape;
    const auto& output_shape = output.shape;
    cycles *=
        (input_shape[1] / output_shape[1]) * (input_shape[2] / output_shape[2]);
  }

  return cycles;
}

}  // namespace

Interpreter::Interpreter(const Model& model, MemoryInterface* memory,
                         Backend* backend)
    : model_(model), memory_(memory), backend_(backend) {
  // Every semaphore starts at zero. The voyager::zeros that says so may sit
  // outside the region being run -- selecting a single loop skips it -- so
  // establish the declared initial state rather than depending on having walked
  // the op that declares it.
  for (const auto& node : model_.semaphore_nodes()) {
    const int64_t slots = semaphore_slots(model_.box(node));
    for (int64_t slot = 0; slot < slots; slot++) {
      backend_->init_semaphore(node, slot, 0);
    }
  }
}

void Interpreter::run(const Model::Selection& selection) {
  for (const auto* op : selection.ops) {
    // Which top-level ops MAX_TILES bounds is the selection's decision -- it
    // stopped the walk at the loop it bounds -- so read it rather than work
    // it out again here. Re-deriving it from TESTS would only agree when a
    // layer's display name happens to also be an op name, and layers.txt
    // names a layer by its extent, not by any one of its ops.
    bound_current_top_ = selection.bounded.count(op) != 0;
    execute_operation(*op);
  }
  bound_current_top_ = true;
}

void Interpreter::execute_region(const voyager::Region& region) {
  for (const auto& op : region.ops()) execute_operation(op);
}

void Interpreter::execute_operation(const voyager::Operation& op) {
  switch (op.op_type_case()) {
    case voyager::Operation::kPrim:
      execute_prim(op, op.prim());
      break;
    case voyager::Operation::kFused:
      // A fusion is always datapath work: its intermediates never leave the
      // pipeline, so there is nothing for the interpreter to do but dispatch
      // it.
      dispatch(op);
      break;
    case voyager::Operation::kLoop:
      if (op.loop().has_for_loop()) {
        execute_for(op, op.loop().for_loop());
      } else if (op.loop().has_while_loop()) {
        execute_while(op, op.loop().while_loop());
      } else {
        throw std::runtime_error("Loop with no body: " + op.name());
      }
      return;  // a loop posts its own semaphore below, via its ops
    case voyager::Operation::kCond:
      execute_cond(op, op.cond());
      return;
    case voyager::Operation::kAsync:
      // A commit carries its own dependency waits and post; it does not use
      // Operation.semaphore, so it returns rather than falling through below.
      execute_async(op, op.async());
      return;
    default:
      throw std::runtime_error("Operation with no op_type: " + op.name());
  }
}

void Interpreter::execute_prim(const voyager::Operation& op,
                               const voyager::PrimOp& prim) {
  const std::string& target = prim.target();

  if (target == "voyager::alloc") {
    // Address assignment is the compiler's; at run time an allocation only
    // establishes its contents. The executable spec zero-fills integer
    // allocations (torch.zeros) -- the sparse flow's running CSR base is
    // read before any write and counts on it -- while float allocations are
    // randn junk that nothing may read before writing.
    if (op.outputs_size() == 1 && op.outputs(0).has_tensor_box()) {
      const auto& box = op.outputs(0).tensor_box();
      const bool is_integer =
          box.dtype().rfind("int", 0) == 0 || box.dtype().rfind("uint", 0) == 0;
      if (is_integer && box.has_memory() && !is_semaphore(box)) {
        // Each slot's own elements, and nothing else. The span between two
        // slots is the pipelining stride, not this buffer's storage: the
        // planner hands that padding to a buffer whose lifetime is disjoint
        // from this one, and zeroing it would clobber that buffer -- and
        // mark its bytes written, which grades a region neither side owns.
        if (backend_->intercepts_data_ops()) {
          backend_->data_op(op, prim, env_);
        } else {
          for (uint32_t bank = 0; bank < banks_of(box); bank++) {
            zero_buffer(to_tensor(box, bank), 1, 0, memory_);
          }
        }
      }
    }
    return;
  }
  if (target == "voyager::zeros") {
    execute_zeros(op, prim);
    return;
  }
  if (target == "voyager::fill") {
    execute_fill(op, prim);
    return;
  }
  if (target == "voyager::async_copy") {
    if (backend_->intercepts_data_ops()) {
      backend_->data_op(op, prim, env_);
    } else {
      run_async_copy(prim, env_, memory_);
    }
    // The completed transfer signals its slot semaphore, post_count times: a
    // tile loaded once but consumed by several commits posts once per consumer.
    const auto sem = prim.kwargs().find("semaphore");
    if (sem != prim.kwargs().end()) {
      const auto& ref = sem->second.tensor_box();
      const int64_t slot = resolve_semaphore_slot(ref, prim.name());
      const auto post_count = prim.kwargs().find("post_count");
      const int64_t count = post_count != prim.kwargs().end()
                                ? eval_int(post_count->second.scalar(), env_)
                                : 1;
      backend_->post_semaphore(ref.box().node(), slot, count);
    }
    return;
  }
  if (target == "voyager::async_wait") {
    execute_async_wait(prim);
    return;
  }
  if (target == "voyager::delinearize_index") {
    const int64_t linear = eval_int(prim.kwargs().at("linear").scalar(), env_);
    const auto basis =
        eval_int_list(prim.kwargs().at("basis").scalar_list(), env_);

    std::vector<int64_t> index(basis.size(), 0);
    int64_t remaining = linear;
    for (int d = static_cast<int>(basis.size()) - 1; d >= 0; d--) {
      index[d] = remaining % basis[d];
      remaining /= basis[d];
    }
    if (op.outputs_size() != static_cast<int>(basis.size())) {
      throw std::runtime_error("delinearize_index " + op.name() +
                               " has the wrong number of results.");
    }
    for (int d = 0; d < op.outputs_size(); d++) {
      env_.define(op.outputs(d).name(), index[d]);
    }
    return;
  }

  // Dispatch on what the operation produces, not on where it runs. An op tagged
  // "cpu" is host-side, but that covers both scalar index arithmetic and tensor
  // work the control processor does itself (a slice, a pad) -- and the latter
  // still needs a kernel, exactly like a datapath op.
  bool produces_only_scalars = op.outputs_size() > 0;
  for (const auto& output : op.outputs()) {
    if (!output.has_scalar()) produces_only_scalars = false;
  }

  if (produces_only_scalars) {
    execute_scalar(op, prim);
  } else {
    dispatch(op);
  }
}

void Interpreter::dispatch(const voyager::Operation& op) {
  // An operation the compiler tagged "cpu" runs on the control processor -- a
  // slice, a host-side pad. It costs the accelerator nothing.
  if (is_datapath(op)) {
    bool is_matrix = false;
    const long cycles = ideal_cycles(op, env_, is_matrix);
    (is_matrix ? matrix_cycles_ : vector_cycles_) += cycles;
  }

  backend_->execute(op, env_);
}

void Interpreter::execute_scalar(const voyager::Operation& op,
                                 const voyager::PrimOp& prim) {
  Scalar result;

  if (strip_namespace(prim.target()) == "_local_scalar_dense") {
    // A data-dependent scalar: the sparse flow reads CSR bookkeeping (an
    // indptr entry) out of memory to size and address the tiles that follow.
    // Read it in the box's own dtype -- these are integer indices.
    const Tensor input = resolve(prim, "input", env_);
    if (get_size(input) != 1) {
      throw std::runtime_error("Scalar operation " + op.name() +
                               " reads more than one element.");
    }

    // to_int64 returns ac_int's Slong, which is long long -- a distinct type
    // from Scalar's int64_t alternative even where both are 64 bits wide.
    // Name the alternative rather than leave the variant's converting
    // assignment to pick one; which conversions it will consider is a point
    // the C++17 compilers here disagree on.
    std::any data = memory_->read_tensor(input);
    if (auto* value =
            std::any_cast<std::shared_ptr<DataTypes::int32[]>>(&data)) {
      result = static_cast<int64_t>((*value)[0].int_val.to_int64());
    } else if (auto* value =
                   std::any_cast<std::shared_ptr<DataTypes::int64[]>>(&data)) {
      result = static_cast<int64_t>((*value)[0].int_val.to_int64());
    } else {
      throw std::runtime_error("Scalar operation " + op.name() +
                               " reads a tensor of non-index dtype " +
                               input.dtype + ".");
    }
  } else if (strip_namespace(prim.target()) == "sym_ite") {
    const bool predicate = to_bool(eval(prim.kwargs().at("b").scalar(), env_));
    result = eval(prim.kwargs().at(predicate ? "t" : "f").scalar(), env_);
  } else {
    const auto input = prim.kwargs().find("input");
    const auto other = prim.kwargs().find("other");
    if (input == prim.kwargs().end() || other == prim.kwargs().end()) {
      throw std::runtime_error("Scalar operation " + op.name() + " (" +
                               strip_namespace(prim.target()) +
                               ") is not binary.");
    }
    result = apply_scalar_op(strip_namespace(prim.target()),
                             eval(input->second.scalar(), env_),
                             eval(other->second.scalar(), env_));
  }

  if (op.outputs_size() != 1) {
    throw std::runtime_error("Scalar operation " + op.name() +
                             " must have exactly one result.");
  }
  env_.define(op.outputs(0).name(), result);
}

void Interpreter::execute_zeros(const voyager::Operation& op,
                                const voyager::PrimOp& prim) {
  if (op.outputs_size() != 1 || !op.outputs(0).has_tensor_box()) {
    throw std::runtime_error("voyager::zeros " + op.name() +
                             " does not declare a buffer.");
  }
  const auto& box = op.outputs(0).tensor_box();
  if (is_semaphore(box)) {
    const int64_t slots = semaphore_slots(box);
    for (int64_t slot = 0; slot < slots; slot++) {
      backend_->init_semaphore(box.node(), slot, 0);
    }
    return;
  }
  if (backend_->intercepts_data_ops()) {
    backend_->data_op(op, prim, env_);
    return;
  }
  zero_buffer(to_tensor(box), banks_of(box), bank_stride_of(box), memory_);
}

// A credit-seeded semaphore: unlike voyager::zeros (which starts a semaphore at
// 0), a fill seeds every slot with `value` credits. That is how an operand
// reused by several consumers is expressed -- one fill posts N at once, and the
// N waits each consume one.
void Interpreter::execute_fill(const voyager::Operation& op,
                               const voyager::PrimOp& prim) {
  if (op.outputs_size() != 1 || !op.outputs(0).has_tensor_box()) {
    throw std::runtime_error("voyager::fill " + op.name() +
                             " does not declare a buffer.");
  }
  const auto& box = op.outputs(0).tensor_box();
  if (!is_semaphore(box)) {
    throw std::runtime_error("voyager::fill " + op.name() +
                             " targets a buffer that is not a semaphore.");
  }
  const int64_t value = eval_int(prim.kwargs().at("value").scalar(), env_);
  const int64_t slots = semaphore_slots(box);
  for (int64_t slot = 0; slot < slots; slot++) {
    backend_->post_semaphore(box.node(), slot, value);
  }
}

int64_t Interpreter::resolve_semaphore_slot(const voyager::TensorBoxRef& ref,
                                            const std::string& op_name) {
  return resolve_bank(ref, env_, op_name);
}

void Interpreter::execute_async_wait(const voyager::PrimOp& prim) {
  const auto& ref = prim.kwargs().at("semaphore").tensor_box();
  const int64_t slot = resolve_semaphore_slot(ref, prim.name());
  // Consume one credit. For a sequential backend this asserts the credit is
  // already there (the walk must reach a wait after its producer); for the
  // Harness it blocks until the completion thread posts.
  backend_->wait_semaphore(ref.box().node(), slot, 1);
}

// A commit: dispatch a region asynchronously. Wait every dependency semaphore,
// run the body, then signal post once the region retires. The walk of the body
// is identical for every backend -- scalars and conditionals resolve here, and
// the datapath/host ops flow through backend->execute(). What differs is the
// backend: a sequential one runs the body in place and end_commit posts
// immediately; the Harness submits the matrix op to the array now (ramp-up) and
// defers the tail + post to its completion thread.
void Interpreter::execute_async(const voyager::Operation& op,
                                const voyager::AsyncOp& async) {
  for (const auto& dep : async.dependencies()) {
    const int64_t slot = resolve_semaphore_slot(dep, op.name());
    backend_->wait_semaphore(dep.box().node(), slot, 1);
  }

  backend_->begin_commit();
  execute_region(async.body());

  if (async.has_post()) {
    const int64_t slot = resolve_semaphore_slot(async.post(), op.name());
    backend_->end_commit(true, async.post().box().node(), slot, 1);
  } else {
    backend_->end_commit(false, "", 0, 0);
  }
}

void Interpreter::execute_for(const voyager::Operation& op,
                              const voyager::ForLoop& loop) {
  const int64_t start = eval_int(loop.start(), env_);
  int64_t end = eval_int(loop.end(), env_);
  const int64_t step = eval_int(loop.step(), env_);
  if (step == 0) throw std::runtime_error("Zero-step loop: " + op.name());

  // MAX_TILES bounds outer tile iterations, which is how regression tests keep
  // very large layers workable. Never apply it to a nested reduction/index
  // loop: truncating the reduction produces an invalid partial output and can
  // leave the software-pipeline drain waiting for a bank that was never posted.
  const int64_t max_tiles = getenv_int("MAX_TILES", 0);
  const bool outermost_loop = loop_depth_ == 0;
  if (max_tiles > 0 && bound_current_top_ && outermost_loop && step > 0) {
    end = std::min(end, start + max_tiles * step);
  }

  std::vector<Scalar> carried;
  for (const auto& arg : loop.iter_args()) {
    carried.push_back(eval(arg.initial(), env_));
  }

  loop_depth_++;
  for (int64_t iv = start; step > 0 ? iv < end : iv > end; iv += step) {
    env_.push();
    env_.define(loop.iv(), iv);
    for (int i = 0; i < loop.iter_args_size(); i++) {
      env_.define(loop.iter_args(i).name(), carried[i]);
    }

    execute_region(loop.body());

    // The body yields the loop-carried state for the next trip. Read it before
    // the scope closes, since the yielded names are defined inside it.
    if (loop.body().yields_size() != loop.iter_args_size()) {
      throw std::runtime_error("Loop " + op.name() +
                               " yields a different number of values than it "
                               "carries.");
    }
    for (int i = 0; i < loop.body().yields_size(); i++) {
      carried[i] = eval(loop.body().yields(i), env_);
    }
    env_.pop();
  }
  loop_depth_--;

  // The final loop-carried values are the loop's results.
  for (int i = 0; i < op.outputs_size() && i < static_cast<int>(carried.size());
       i++) {
    env_.define(op.outputs(i).name(), carried[i]);
  }
}

void Interpreter::execute_while(const voyager::Operation& op,
                                const voyager::WhileLoop& loop) {
  std::vector<Scalar> carried;
  for (const auto& arg : loop.iter_args()) {
    carried.push_back(eval(arg.initial(), env_));
  }

  const int64_t max_tiles = getenv_int("MAX_TILES", 0);
  const bool outermost_loop = loop_depth_ == 0;
  int64_t trips = 0;
  loop_depth_++;
  while (true) {
    if (max_tiles > 0 && bound_current_top_ && outermost_loop &&
        trips >= max_tiles) {
      break;
    }

    env_.push();
    for (int i = 0; i < loop.iter_args_size(); i++) {
      env_.define(loop.iter_args(i).name(), carried[i]);
    }

    execute_region(loop.condition());
    if (loop.condition().yields_size() != 1) {
      throw std::runtime_error("While loop " + op.name() +
                               " must yield exactly one predicate.");
    }
    if (!to_bool(eval(loop.condition().yields(0), env_))) {
      env_.pop();
      break;
    }

    execute_region(loop.body());
    for (int i = 0; i < loop.body().yields_size(); i++) {
      carried[i] = eval(loop.body().yields(i), env_);
    }
    env_.pop();
    trips++;
  }
  loop_depth_--;

  for (int i = 0; i < op.outputs_size() && i < static_cast<int>(carried.size());
       i++) {
    env_.define(op.outputs(i).name(), carried[i]);
  }
}

void Interpreter::execute_cond(const voyager::Operation& op,
                               const voyager::CondOp& cond) {
  const bool predicate = to_bool(eval(cond.predicate(), env_));
  const voyager::Region& region =
      predicate ? cond.true_region() : cond.false_region();

  env_.push();
  execute_region(region);

  // A conditional that produces scalars binds them in the enclosing scope.
  std::vector<Scalar> yielded;
  for (const auto& yield : region.yields())
    yielded.push_back(eval(yield, env_));
  env_.pop();

  for (int i = 0; i < op.outputs_size() && i < static_cast<int>(yielded.size());
       i++) {
    env_.define(op.outputs(i).name(), yielded[i]);
  }
}
