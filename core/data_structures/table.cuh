#ifndef HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_
#define HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_

#include <rapidcsv.h>

#include <cstring>

#include "core/util/atomic.h"

namespace sics::hyper_blocker::core::data_structures {

class Table {
 public:
  Table(const std::string& data_path, const std::string& sep = ",",
        bool read_header = false) {
    // Read the data from the file.
    rapidcsv::Document doc(data_path, rapidcsv::LabelParams(0, -1));

    n_rows_ = doc.GetRowCount();
    n_cols_ = doc.GetColumnCount();
    col_size_ = new size_t[n_cols_]();
    col_offset_ = new size_t[n_cols_]();

    // First traversal to get the max length of the strings.
    auto max_length_col = new unsigned[n_cols_]();
    for (size_t i = 0; i < n_cols_; i++) {
      auto&& col = doc.GetColumn<std::string>(i);
      std::for_each(col.begin(), col.end(),
                    [&max_length_col, i](std::string& s) {
                      sics::hyper_blocker::core::util::WriteMax(
                          (max_length_col + i), (unsigned)s.length());
                    });
      col_size_[i] = max_length_col[i] + 1;
    }

    // Compute mata data of the table.
    for (size_t i = 0; i < n_cols_; i++) {
      aligned_tuple_size_ += col_size_[i];
      if (i > 0) col_offset_[i] = col_offset_[i - 1] + col_size_[i - 1];
    }

    aligned_tuple_size_ = (((aligned_tuple_size_ + 1) >> 6) << 6) + 64;
    data_ = new char[aligned_tuple_size_ * n_rows_ * sizeof(char)]();

    // Second traversal to fill the data.
    for (size_t i = 0; i < n_cols_; i++) {
      auto&& col = doc.GetColumn<std::string>(i);
      for (size_t j = 0; j < n_rows_; j++) {
        std::memcpy(data_ + aligned_tuple_size_ * j + col_offset_[i],
                    col[j].c_str(), col[j].length() * sizeof(char));
      }
    }

    delete[] max_length_col;
  }

  void ShowTable() {
    std::cout << "ShowTable: " << std::endl;
    for (size_t j = 0; j < n_rows_; j++) {
      for (size_t i = 0; i < n_cols_; i++) {
        std::cout << (data_ + aligned_tuple_size_ * j + col_offset_[i]) << " ";
      }
      std::cout << std::endl;
    }
  }

  char* get_data() const { return data_; }

  size_t get_aligned_tuple_size() const { return aligned_tuple_size_; }

  size_t get_n_rows() const { return n_rows_; }

  size_t get_n_cols() const { return n_cols_; }

 public:
  size_t n_rows_ = 0;
  size_t n_cols_ = 0;
  size_t aligned_tuple_size_ = 0;
  size_t* col_size_;
  size_t* col_offset_;
  char* data_;
};

}  // namespace sics::hyper_blocker::core::data_structures

#endif  // HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_
