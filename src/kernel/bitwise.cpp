#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    // Implement your version...
    const std::size_t n = std::min({result.size(), a.size(), b.size()});

    auto *r = result.data();
    const auto *pa = a.data();
    const auto *pb = b.data();

    /* 
     * Simplified bit by bit
     * each bit:
     *  result = (either & 0x18) | (~either & 0x81) | 0x24
     * where either = a/b
     * 
     * repeat the mask across a 64-bit word
     */

    constexpr std::uint64_t kEitherMask = 0x1818181818181818ULL;
    constexpr std::uint64_t kNotEitherMask = 0x8181818181818181ULL;
    constexpr std::uint64_t kConstMask = 0x2424242424242424ULL;

    std::size_t i = 0;

    // Process 32 bytes per loop iteration: 4 packed uint64_t blocks.
    for (; i + 31 < n; i += 32) {
        std::uint64_t a0, a1, a2, a3;
        std::uint64_t b0, b1, b2, b3;

        std::memcpy(&a0, pa + i + 0,  sizeof(a0));
        std::memcpy(&b0, pb + i + 0,  sizeof(b0));
        std::memcpy(&a1, pa + i + 8,  sizeof(a1));
        std::memcpy(&b1, pb + i + 8,  sizeof(b1));
        std::memcpy(&a2, pa + i + 16, sizeof(a2));
        std::memcpy(&b2, pb + i + 16, sizeof(b2));
        std::memcpy(&a3, pa + i + 24, sizeof(a3));
        std::memcpy(&b3, pb + i + 24, sizeof(b3));

        const std::uint64_t e0 = a0 | b0;
        const std::uint64_t e1 = a1 | b1;
        const std::uint64_t e2 = a2 | b2;
        const std::uint64_t e3 = a3 | b3;

        const std::uint64_t o0 =
            (e0 & kEitherMask) | (~e0 & kNotEitherMask) | kConstMask;
        const std::uint64_t o1 =
            (e1 & kEitherMask) | (~e1 & kNotEitherMask) | kConstMask;
        const std::uint64_t o2 =
            (e2 & kEitherMask) | (~e2 & kNotEitherMask) | kConstMask;
        const std::uint64_t o3 =
            (e3 & kEitherMask) | (~e3 & kNotEitherMask) | kConstMask;

        std::memcpy(r + i + 0,  &o0, sizeof(o0));
        std::memcpy(r + i + 8,  &o1, sizeof(o1));
        std::memcpy(r + i + 16, &o2, sizeof(o2));
        std::memcpy(r + i + 24, &o3, sizeof(o3));
    }

    // Process remaining 8-byte blocks.
    for (; i + 7 < n; i += 8) {
        std::uint64_t av;
        std::uint64_t bv;

        std::memcpy(&av, pa + i, sizeof(av));
        std::memcpy(&bv, pb + i, sizeof(bv));

        const std::uint64_t e = av | bv;
        const std::uint64_t out =
            (e & kEitherMask) | (~e & kNotEitherMask) | kConstMask;

        std::memcpy(r + i, &out, sizeof(out));
    }

    // Scalar tail.
    for (; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(pa[i]);
        const auto ub = static_cast<std::uint8_t>(pb[i]);

        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto out = static_cast<std::uint8_t>(
            (either & 0x18u) | ((~either) & 0x81u) | 0x24u
        );

        r[i] = static_cast<std::int8_t>(out);
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}
