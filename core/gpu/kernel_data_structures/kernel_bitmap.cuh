#ifndef HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_
#define HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_

namespace sics {
namespace hyperblocker {
namespace core {
namespace gpu {

#define WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

// @DESCRIPTION
//
// Bitmap is a mapping from integers~(indexes) to bits. If the unit is
// occupied, the bit is a nonzero integer constant and if it is empty, the bit
// is zero.
__device__ class KernelBitmap {
public:
  __device__ KernelBitmap(size_t size) : size_(size) { Init(size); }

  __device__ ~KernelBitmap() { free(data_); }

  __device__ void Init(size_t size) {
    free(data_);
    size_ = size;
    data_ = (uint64_t *)malloc(sizeof(uint64_t) * WORD_OFFSET(size_) + 1);
  }

  __device__ void Clear() {
    size_t bm_size = WORD_OFFSET(size_);
    for (size_t i = 0; i <= bm_size; i++)
      data_[i] = 0;
  }

  __device__ void Fill() {
    size_t bm_size = WORD_OFFSET(size_);
    for (size_t i = 0; i < bm_size; i++) {
      data_[i] = 0xffffffffffffffff;
    }
    data_[bm_size] = 0;
    for (size_t i = (bm_size << 6); i < size_; i++) {
      data_[bm_size] |= 1ul << BIT_OFFSET(i);
    }
  }

  __device__ bool GetBit(size_t i) const {
    if (i > size_)
      return 0;
    return data_[WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
  }

  __device__ void SetBit(size_t i) {
    if (i > size_)
      return;
    *(data_ + WORD_OFFSET(i)) =
        *((data_ + WORD_OFFSET(i))) | (1ul << BIT_OFFSET(i));
  }

  __device__ size_t Count() const {
    size_t count = 0;
    for (size_t i = 0; i <= WORD_OFFSET(size_); i++) {
      auto x = data_[i];
      x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
      x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
      x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
      x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
      x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
      x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
      count += (size_t)x;
    }
    return count;
  };

  __device__ size_t get_size() const { return size_; }

private:
  size_t size_;
  uint64_t *data_ = nullptr;
};

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // INC_51_11_ER_HYPERBLOCKER_CORE_GPU_KERNEL_DATA_STRUCTURES_KERNEL_BITMAP_CUH_
