#include "relu.h"
#include <algorithm>
#include <cstdint>
#include <random>

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

void stu_relu(std::span<float> data) {
    // TODO: Implement your version, and call it in stu_relu_wrapper
    float *p = data.data();
    const size_t n = data.size();

    constexpr float zero = 0.0f;
    size_t i = 0;

    // Unrolling
    for(; i + 15 < n; i += 16) {
        const float x0 = p[i];
        const float x1 = p[i + 1];
        const float x2 = p[i + 2];
        const float x3 = p[i + 3];
        const float x4 = p[i + 4];
        const float x5 = p[i + 5];
        const float x6 = p[i + 6];
        const float x7 = p[i + 7];
        const float x8 = p[i + 8];
        const float x9 = p[i + 9];
        const float x10 = p[i + 10];
        const float x11 = p[i + 11];
        const float x12 = p[i + 12];
        const float x13 = p[i + 13];
        const float x14 = p[i + 14]; 
        const float x15 = p[i + 15];

        p[i] = x0 < zero ? zero : x0;
        p[i + 1] = x1 < zero ? zero : x1;
        p[i + 2] = x2 < zero ? zero : x2;
        p[i + 3] = x3 < zero ? zero : x3;
        p[i + 4] = x4 < zero ? zero : x4;
        p[i + 5] = x5 < zero ? zero : x5;
        p[i + 6] = x6 < zero ? zero : x6;
        p[i + 7] = x7 < zero ? zero : x7;
        p[i + 8] = x8 < zero ? zero : x8;
        p[i + 9] = x9 < zero ? zero : x9;
        p[i + 10] = x10 < zero ? zero : x10;
        p[i + 11] = x11 < zero ? zero : x11;
        p[i + 12] = x12 < zero ? zero : x12;
        p[i + 13] = x13 < zero ? zero : x13;
        p[i + 14] = x14 < zero ? zero : x14;
        p[i + 15] = x15 < zero ? zero : x15;
    }

    // Remaining
    for (; i < n; i++) {
        const float x = p[i];
        p[i] = x < zero ? zero : x;
    }
}

void naive_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    naive_relu(args.data);
}

void stu_relu_wrapper(void *ctx) {
    auto &args = *static_cast<relu_args *>(ctx);
    stu_relu(args.data);
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
