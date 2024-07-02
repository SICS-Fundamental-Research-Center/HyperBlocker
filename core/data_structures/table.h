#ifndef HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_
#define HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_

#include <cstring>
#include <cuda_runtime.h>

#include <rapidcsv.h>

#include "core/util/atomic.h"

namespace sics {
namespace hyperblocker {
namespace core {
namespace data_structures {

class SerializedTable {
public:
  void Show(size_t n_rows = 100) const {
    n_rows = n_rows < n_rows_ ? n_rows : n_rows_;
    std::cout << "\nShowTable " << n_rows << " rows : " << std::endl;
    for (size_t j = 0; j < n_rows; j++) {
      for (size_t i = 0; i < n_cols_; i++) {
        std::cout << (data_ + aligned_tuple_size_ * j + col_offset_[i]) << "|";
      }
      std::cout << std::endl;
    }
    std::cout << "==========================" << std::endl;
  }

  char *get_data_base_ptr() const { return data_; }
  size_t *get_col_size_base_ptr() const { return col_size_; }
  size_t *get_col_offset_base_ptr() const { return col_offset_; }

  char *get_term_ptr(size_t col, size_t row) const {
    return data_ + aligned_tuple_size_ * row + col_offset_[col];
  }

  size_t get_aligned_tuple_size() const { return aligned_tuple_size_; }
  size_t get_n_rows() const { return n_rows_; }
  size_t get_n_cols() const { return n_cols_; }

private:
  friend class SerializableTable;

  size_t n_rows_ = 0;
  size_t n_cols_ = 0;
  size_t aligned_tuple_size_ = 0;
  size_t *col_size_;
  size_t *col_offset_;
  char *data_;
};

class SerializableTable {
public:
  SerializableTable(const std::vector<std::vector<std::string>> &cols)
      : cols_(cols) {
    n_cols_ = cols.size();
    n_rows_ = cols[0].size();
    max_length_per_col_ = new unsigned[n_cols_]();
    min_length_per_col_ = new unsigned[n_cols_]();
    avg_length_per_col_ = new unsigned[n_cols_]();
    memset(min_length_per_col_, 1, sizeof(unsigned) * n_cols_);

    Serialize2PinnedMemory();
  }

  void PrintStatistics() const {
    for (size_t i = 0; i < n_cols_; i++) {
      std::cout << "col " << i << " max_length: " << max_length_per_col_[i]
                << " min_length: " << min_length_per_col_[i]
                << " avg_length: " << avg_length_per_col_[i] << std::endl;
    }
  }

  void Serialize2PinnedMemory() {
    if (is_complete_)
      return;

    serialized_table_.n_cols_ = n_cols_;
    serialized_table_.n_rows_ = n_rows_;

    cudaHostAlloc(&(serialized_table_.col_size_), sizeof(size_t) * n_cols_,
                  cudaHostAllocPortable);
    cudaHostAlloc(&(serialized_table_.col_offset_), sizeof(size_t) * n_cols_,
                  cudaHostAllocPortable);

    for (size_t i = 0; i < n_cols_; i++) {
      auto &&col = cols_[i];
      std::for_each(col.begin(), col.end(), [&, i](std::string &s) {
        WriteMax((max_length_per_col_ + i), (unsigned)s.length());
        WriteMin((min_length_per_col_ + i), (unsigned)s.length());
        WriteAdd((avg_length_per_col_ + i), (unsigned)s.length());
      });
      serialized_table_.col_size_[i] = max_length_per_col_[i] + 1;
    }

    for (size_t i = 0; i < n_cols_; i++)
      avg_length_per_col_[i] /= n_rows_;

    // Compute mata data of the table.
    for (size_t i = 0; i < n_cols_; i++) {
      serialized_table_.aligned_tuple_size_ += serialized_table_.col_size_[i];
      if (i > 0)
        serialized_table_.col_offset_[i] =
            serialized_table_.col_offset_[i - 1] +
            serialized_table_.col_size_[i - 1];
    }

    serialized_table_.aligned_tuple_size_ =
        (((serialized_table_.aligned_tuple_size_ + 1) >> 6) << 6) + 64;

    cudaHostAlloc(&(serialized_table_.data_),
                  sizeof(char) * n_rows_ *
                      serialized_table_.aligned_tuple_size_,
                  cudaHostAllocPortable);

    // Second traversal to fill the data.
    for (size_t i = 0; i < n_cols_; i++) {
      auto &&col = cols_[i];
      for (size_t j = 0; j < n_rows_; j++) {
        std::memcpy(serialized_table_.data_ +
                        serialized_table_.aligned_tuple_size_ * j +
                        serialized_table_.col_offset_[i],
                    col[j].c_str(), col[j].length() * sizeof(char));
      }
    }
    is_complete_ = true;
  }

  SerializedTable GetSerializedTable() {
    if (is_complete_)
      return serialized_table_;
    else {
      Serialize2PinnedMemory();
      return serialized_table_;
    }
  }

  size_t get_n_rows() const { return n_rows_; }
  size_t get_n_cols() const { return n_cols_; }

  std::vector<std::vector<std::string>> get_cols() const { return cols_; }

private:
  bool is_complete_ = false;

  SerializedTable serialized_table_;

  size_t n_cols_ = 0;
  size_t n_rows_ = 0;

  unsigned *avg_length_per_col_ = nullptr;
  unsigned *max_length_per_col_ = nullptr;
  unsigned *min_length_per_col_ = nullptr;

  std::vector<std::vector<std::string>> cols_;
};

} // namespace data_structures
} // namespace core
} // namespace hyperblocker
} // namespace sics

#endif // HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_