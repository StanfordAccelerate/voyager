#pragma once

#include <string>
#include <vector>

#include "test/common/ArrayMemory.h"
#include "test/common/Model.h"
#include "test/common/Utils.h"

// Loads a buffer's contents from the compiler's float32 dump, through the
// dtype's packing, at the address the graph gives it.
void load_tensor(const Tensor& tensor, const std::string& data_dir,
                 MemoryInterface* memory);

// Compares what two simulators left in memory element-wise with
// torch.isclose semantics and reports how many elements are not close,
// printing "Error count: N".
//
// Both DRAM and scratchpad buffers are graded, which is what catches a store
// that lands outside its own allocation. A run bounded by MAX_TILES fills a
// scratchpad tile completely but only a prefix of the DRAM output, so the two
// are usually graded to very different depths; the write masks keep that
// honest by comparing only what the run produced.
//
// Each diff file is named "<prefix><tensor>.<first>_vs_<second>.txt".
int compare_memories(const std::vector<const voyager::TensorBox*>& expected,
                     const Model& model, ArrayMemory* first,
                     const std::string& first_name, ArrayMemory* second,
                     const std::string& second_name, const std::string& out_dir,
                     const std::string& prefix, float rtol, float atol,
                     bool require_complete = false);

// Element-wise comparison of two runs' contents of one buffer, with the
// difference histogram compare_memories reports. Only Checker uses these.

static void log_diff_buckets(const std::string& label, const int buckets[5],
                             size_t size) {
  spdlog::info("{}:\n", label);
  spdlog::info("< 0.001: {:8d} ({:6.2f}%)\n", buckets[0],
               static_cast<float>(buckets[0]) / size * 100.0f);
  spdlog::info("< 0.01:  {:8d} ({:6.2f}%)\n", buckets[1],
               static_cast<float>(buckets[1]) / size * 100.0f);
  spdlog::info("< 0.1:   {:8d} ({:6.2f}%)\n", buckets[2],
               static_cast<float>(buckets[2]) / size * 100.0f);
  spdlog::info("< 1:     {:8d} ({:6.2f}%)\n", buckets[3],
               static_cast<float>(buckets[3]) / size * 100.0f);
  spdlog::info("> 1:     {:8d} ({:6.2f}%)\n", buckets[4],
               static_cast<float>(buckets[4]) / size * 100.0f);
}

template <typename T1, typename T2>
long compare_arrays(std::any matrix_a, const std::string& name_a,
                    std::any matrix_b, const std::string& name_b, size_t size,
                    const std::string& filename,
                    const std::vector<uint8_t>& valid, float rtol, float atol) {
  spdlog::info("Comparing {} and {} (size: {}) -> output: {}\n", name_a, name_b,
               size, filename);

  std::ofstream diff_file(filename);
  std::ostringstream buffer;

  buffer << name_a << " vs. " << name_b << '\n';
  constexpr size_t flush_interval = 1000;
  constexpr size_t max_diff_lines = 10000;

  int abs_diff_buckets[5] = {0};
  int rel_diff_buckets[5] = {0};
  double sum_abs = 0.0;
  size_t nonfinite = 0;
  size_t diff_lines = 0;
  size_t omitted_diff_lines = 0;

  T1* matrix_a_ptr = std::any_cast<T1*>(matrix_a);
  T2* matrix_b_ptr = std::any_cast<T2*>(matrix_b);

  size_t num_valid = 0;
  long mismatched = 0;
  float max_abs_diff = 0.0f;
  size_t max_abs_index = 0;
  float max_rel_diff = 0.0f;
  size_t max_rel_index = 0;

  for (size_t i = 0; i < size; ++i) {
    // Skip anything the run never produced (a partial run under MAX_TILES).
    if (!valid[i]) continue;
    num_valid++;

    float a = static_cast<float>(matrix_a_ptr[i]);
    float b = static_cast<float>(matrix_b_ptr[i]);
    sum_abs += std::abs(a) + std::abs(b);
    const float abs_diff = std::abs(a - b);
    const bool finite = std::isfinite(a) && std::isfinite(b);

    // torch.isclose, with the second simulator as the reference: equal values
    // (including same-sign infinities) are close, any other non-finite pair is
    // not, and finite pairs must satisfy |a - b| <= atol + rtol * |b|.
    const bool close =
        a == b || (finite && abs_diff <= atol + rtol * std::abs(b));

    float normalized = 0.0f;
    if (!finite) {
      nonfinite++;
      normalized = std::numeric_limits<float>::infinity();
    } else {
      const float denominator = (std::abs(a) + std::abs(b)) / 2.0f;
      normalized = (a == 0.0f && b == 0.0f) ? 0.0f : abs_diff / denominator;
    }

    if (!close) {
      mismatched++;
      if (abs_diff > max_abs_diff) {
        max_abs_diff = abs_diff;
        max_abs_index = i;
      }
      if (normalized > max_rel_diff) {
        max_rel_diff = normalized;
        max_rel_index = i;
      }

      // A full LLM tensor has tens of millions of values. Keep a bounded
      // sample of the failing pairs and always compute the aggregate
      // statistics over the complete tensor.
      if (diff_lines < max_diff_lines) {
        buffer << "[" << i << "] " << a << " vs. " << b << '\n';
        diff_lines++;
        if (diff_lines % flush_interval == 0) {
          diff_file << buffer.str();
          buffer.str("");
          buffer.clear();
        }
      } else {
        omitted_diff_lines++;
      }
    }

    abs_diff_buckets[0] += abs_diff < 0.001f;
    abs_diff_buckets[1] += abs_diff < 0.01f;
    abs_diff_buckets[2] += abs_diff < 0.1f;
    abs_diff_buckets[3] += abs_diff < 1.0f;
    abs_diff_buckets[4] += abs_diff >= 1.0f || !finite;

    rel_diff_buckets[0] += normalized < 0.001f;
    rel_diff_buckets[1] += normalized < 0.01f;
    rel_diff_buckets[2] += normalized < 0.1f;
    rel_diff_buckets[3] += normalized < 1.0f;
    rel_diff_buckets[4] += normalized >= 1.0f;
  }

  diff_file << buffer.str();
  if (omitted_diff_lines > 0) {
    diff_file << "... " << omitted_diff_lines
              << " additional differences omitted\n";
  }
  diff_file.close();

  if (num_valid == 0) {
    spdlog::warn("WARNING: the run produced none of this buffer!\n");
    return 1;
  }

  spdlog::info("Outputs compared: {} / {} ({}%)\n", num_valid, size,
               (100.0 * num_valid / size));
  spdlog::info("-------------------------------\n");
  log_diff_buckets("Absolute Difference Count", abs_diff_buckets, num_valid);
  spdlog::info("-------------------------------\n");
  log_diff_buckets("Relative Difference Count", rel_diff_buckets, num_valid);
  spdlog::info("-------------------------------\n");

  if (sum_abs == 0.0) {
    spdlog::warn("WARNING: All compared values are zero!\n");
  }

  spdlog::info("Mismatched elements: {} / {} ({:.2f}%)\n", mismatched,
               num_valid, 100.0 * mismatched / num_valid);
  if (mismatched > 0) {
    spdlog::info("Greatest absolute difference: {} at [{}]\n", max_abs_diff,
                 max_abs_index);
    spdlog::info("Greatest relative difference: {} at [{}]\n", max_rel_diff,
                 max_rel_index);
  }
  if (nonfinite > 0) {
    spdlog::error("Non-finite values compared: {}\n", nonfinite);
  }
  return mismatched;
}

template <typename T>
bool compare_arrays_helper(const Tensor& tensor, const std::any& output1,
                           const std::string& name1, const std::any& output2,
                           const std::string& name2,
                           const std::string& filename,
                           const std::vector<uint8_t>& valid, float rtol,
                           float atol, long& mismatched) {
  if (tensor.dtype == DataTypes::TypeName<T>::name()) {
    mismatched +=
        compare_arrays<T, T>(output1, name1, output2, name2, get_size(tensor),
                             filename, valid, rtol, atol);
    return true;
  }
  return false;
}

template <typename... Ts>
long compare_arrays(const Tensor& tensor, const std::any& output1,
                    const std::string& name1, const std::any& output2,
                    const std::string& name2, const std::string& filename,
                    const std::vector<uint8_t>& valid, float rtol, float atol) {
  // Integer codes are graded in code units: under a tolerant profile a
  // single-LSB disagreement is rounding noise, not an error.
  if (atol > 0.0f && (tensor.dtype.rfind("int", 0) == 0 ||
                      tensor.dtype.rfind("uint", 0) == 0)) {
    atol = std::max(atol, 1.0f);
  }

  long mismatched = 0;
  bool matched =
      (compare_arrays_helper<Ts>(tensor, output1, name1, output2, name2,
                                 filename, valid, rtol, atol, mismatched) ||
       ...);

  if (!matched) {
    throw std::runtime_error("Unsupported tensor dtype: " + tensor.dtype);
  }

  return mismatched;
}
