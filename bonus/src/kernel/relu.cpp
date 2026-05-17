#include "relu.h"
#include <algorithm>
#include <cstdint>
#include <array>
#include <random>
#include <thread>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

void initialize_relu(relu_args *args, const size_t size,
                     const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr float mean = 0.0f;
    constexpr float stddev = 1.0f;

    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);

    args->data.resize(size);

    for (auto &value : args->data) {
        value = dist(gen);
    }
}

void naive_relu(std::span<float> data) {
    for (auto &&value : data) {
        if (value < 0.0f) {
            value = 0.0f;
        }
    }
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
__attribute__((target("avx2"), optimize("O3")))
#endif
void stu_relu(std::span<float> data) {
    float* values = data.data();
    const std::size_t n = data.size();

    std::size_t i = 0;
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        const __m256 v0 = _mm256_loadu_ps(values + i);
        const __m256 v1 = _mm256_loadu_ps(values + i + 8);
        const __m256 v2 = _mm256_loadu_ps(values + i + 16);
        const __m256 v3 = _mm256_loadu_ps(values + i + 24);

        _mm256_storeu_ps(values + i, _mm256_max_ps(v0, zero));
        _mm256_storeu_ps(values + i + 8, _mm256_max_ps(v1, zero));
        _mm256_storeu_ps(values + i + 16, _mm256_max_ps(v2, zero));
        _mm256_storeu_ps(values + i + 24, _mm256_max_ps(v3, zero));
    }

    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(values + i);
        _mm256_storeu_ps(values + i, _mm256_max_ps(v, zero));
    }
#endif

    for (; i < n; ++i) {
        values[i] = std::max(0.0f, values[i]);
    }
}

void naive_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    naive_relu(args.data);
}

void stu_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    constexpr std::size_t kThreadCount = 4;
    const std::size_t n = args.data.size();

    if (n >= kThreadCount * 65536) {
        std::array<std::thread, kThreadCount - 1> workers;
        for (std::size_t t = 0; t + 1 < kThreadCount; ++t) {
            const std::size_t begin = (n * t) / kThreadCount;
            const std::size_t end = (n * (t + 1)) / kThreadCount;
            workers[t] = std::thread([&, begin, end] {
                stu_relu(std::span<float>(args.data.data() + begin, end - begin));
            });
        }

        const std::size_t main_begin = (n * (kThreadCount - 1)) / kThreadCount;
        stu_relu(std::span<float>(args.data.data() + main_begin, n - main_begin));
        for (auto& worker : workers) {
            worker.join();
        }
    } else {
        stu_relu(args.data);
    }
}

bool relu_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<relu_args *>(stu_ctx);
    auto &ref_args = *static_cast<relu_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;

    if (stu_args.data.size() != ref_args.data.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.data.size(),
                  ref_args.data.size());
        return false;
    }

    double max_rel = 0.0;
    size_t worst_i = 0;
    const double atol = 1e-6;

    for (size_t i = 0; i < ref_args.data.size(); ++i) {
        const double r = static_cast<double>(ref_args.data[i]);
        const double s = static_cast<double>(stu_args.data[i]);
        const double err = std::abs(s - r);
        const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;

        if (rel > max_rel) {
            max_rel = rel;
            worst_i = i;
        }

        if (err > (atol + eps * std::abs(r))) {
            debug_log("\tDEBUG: fail at {}: ref={} stu={} err={} rel={} thr={}\n",
                      i,
                      ref_args.data[i],
                      stu_args.data[i],
                      err,
                      rel,
                      (atol + eps * std::abs(r)));
            return false;
        }
    }

    debug_log("\tDEBUG: relu_check passed. max_rel={} at i={}\n",
              max_rel,
              worst_i);
    return true;
}
