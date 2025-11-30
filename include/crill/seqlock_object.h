// crill - the Cross-platform Real-time, I/O, and Low-Latency Library
// Copyright (c) 2022 - Timur Doumler and Fabian Renn-Giles
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

#ifndef CRILL_SEQLOCK_OBJECT_H
#define CRILL_SEQLOCK_OBJECT_H

#include <array>
#include <atomic>
#include <crill/bytewise_atomic_memcpy.h>

namespace crill {

// A portable C++ implementation of a seqlock wrapping a single value of trivially
// copyable type, as proposed in P3825.
// This implementation is inspired by Hans Boehm's paper
// "Can Seqlocks Get Along With Programming Language Memory Models?"
// and the C implementation in jemalloc.
//
// This version allows only a single writer. Writes are guaranteed wait-free.
// It also allows multiple concurrent readers, which are wait-free against
// each other, but can block if there is a concurrent write.
//
// On platforms that provide std::atomic_ref or an equivalent compiler intrinsic,
// this implementation internally uses crill::atomic_load_per_byte_memcpy and
// crill::atomic_store_per_byte_memcpy. Otherwise, it falls back to performing
// per-byte atomic loads and stores.
template <typename T>
class seqlock_object
{
public:
    static_assert(std::is_trivially_copyable_v<T>);

    // Creates a seqlock_object with a default-constructed value.
    seqlock_object()
    {
        store(T());
    }

    // Creates a seqlock_object with the given value.
    seqlock_object(T t)
    {
        store(t);
    }

    // Reads and returns the current value.
    // Non-blocking guarantees: wait-free if there are no concurrent writes,
    // otherwise none.
    // Note: Instead of `load`, you may want to use the more efficient
    // `crill::progressive_backoff_wait([&t]{ return try_load(t); })
    T load() const noexcept
    {
        T t;
        while (!try_load(t)) /* keep trying */;
        return t;
    }

    // Attempts to read the current value and write it into the passed-in object.
    // Returns: true if the read succeeded, false otherwise.
    // Non-blocking guarantees: wait-free.
    bool try_load(T& t) const noexcept
    {
        std::size_t seq1 = seq.load(std::memory_order_acquire);
        if (seq1 % 2 != 0)
            return false;

        atomic_storage.memcpy_out(t);
        std::size_t seq2 = seq.load(std::memory_order_relaxed);
        return seq1 == seq2;
    }

    // Updates the current value to the value passed in.
    // Non-blocking guarantees: wait-free.
    void store(T t) noexcept
    {
        std::size_t old_seq = seq.load(std::memory_order_relaxed);
        seq.store(old_seq + 1, std::memory_order_relaxed);
        // Note: seq.load + store usually has better performance characteristics than seq.fetch_add(1)

        atomic_storage.memcpy_in(t);
        seq.store(old_seq + 2, std::memory_order_release);
    }

private:
    std::atomic<std::size_t> seq = 0;
    static_assert(decltype(seq)::is_always_lock_free);

    struct atomic_storage_t {
        T memcpy_out(T& value_out) const {
          #if CRILL_BYTEWISE_ATOMIC_MEMCPY_AVAILABLE
            crill::atomic_load_per_byte_memcpy(&value_out, &data, sizeof(data), std::memory_order_acquire);
          #else
            char* bytes_out = reinterpret_cast<char*>(&value_out);

            for (std::size_t i = 0; i < sizeof(T); ++i)
                bytes_out[i] = data[i].load(std::memory_order_relaxed);

            std::atomic_thread_fence(std::memory_order_acquire);
          #endif
        }

        void memcpy_in(T& value_in) {
          #if CRILL_BYTEWISE_ATOMIC_MEMCPY_AVAILABLE
            crill::atomic_store_per_byte_memcpy(&data, &value_in, sizeof(data), std::memory_order_release);
          #else
            const char* bytes_in_ptr = reinterpret_cast<const char*>(&value_in);

            std::atomic_thread_fence(std::memory_order_release);

            for (std::size_t i = 0; i < sizeof(T); ++i)
                data[i].store(bytes_in_ptr[i], std::memory_order_relaxed);
          #endif
        }

    private:
      #if CRILL_BYTEWISE_ATOMIC_MEMCPY_AVAILABLE
        char data[sizeof(T)];
      #else
        alignas(T) std::atomic<uint8_t> data[sizeof(T)];
      #endif
    };

    atomic_storage_t atomic_storage;
};

} // namespace crill

#endif //CRILL_SEQLOCK_OBJECT_H
