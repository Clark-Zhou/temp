#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

void initialize_matmul(matmul_args& args, int n, uint32_t seed) {
    if (n <= 0) {
        throw std::invalid_argument("initialize_matmul: n must be positive.");
    }

    args.n = n;
    args.epsilon = 1e-3;

    const size_t elem_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    args.A.resize(elem_count);
    args.B.resize(elem_count);
    args.C.assign(elem_count, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elem_count; ++i) {
        args.A[i] = dist(rng);
        args.B[i] = dist(rng);
    }
}

void naive_matmul(std::vector<float>& C,
                  const std::vector<float>& A,
                  const std::vector<float>& B,
                  int n) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n) {
    const std::size_t nn =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    if (C.size() != nn) {
        C.resize(nn);
    }
    if (C.size() != nn) {
        C.resize(nn);
    }

    static std::vector<float> BT;
    if (BT.size() != nn) {
        BT.resize(nn);
    }

#if defined(BONUS_USE_OPENMP) && BONUS_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        const float* b_row = B.data() + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) {
            BT[static_cast<std::size_t>(j) * n + i] = b_row[j];
        }
    }

    const float* __restrict__ a = A.data();
    const float* __restrict__ bt = BT.data();
    float* __restrict__ c = C.data();

#if defined(BONUS_USE_OPENMP) && BONUS_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        const float* a_row = a + static_cast<std::size_t>(i) * n;
        float* c_row = c + static_cast<std::size_t>(i) * n;

        int j = 0;

        for (; j + 7 < n; j += 8) {
            const float* bt0 = bt + static_cast<std::size_t>(j) * n;
            const float* bt1 = bt0 + n;
            const float* bt2 = bt1 + n;
            const float* bt3 = bt2 + n;
            const float* bt4 = bt3 + n;
            const float* bt5 = bt4 + n;
            const float* bt6 = bt5 + n;
            const float* bt7 = bt6 + n;

            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            float s4 = 0.0f;
            float s5 = 0.0f;
            float s6 = 0.0f;
            float s7 = 0.0f;

            for (int k = 0; k < n; ++k) {
                const float av = a_row[k];

                s0 += av * bt0[k];
                s1 += av * bt1[k];
                s2 += av * bt2[k];
                s3 += av * bt3[k];
                s4 += av * bt4[k];
                s5 += av * bt5[k];
                s6 += av * bt6[k];
                s7 += av * bt7[k];
            }

            c_row[j]     = s0;
            c_row[j + 1] = s1;
            c_row[j + 2] = s2;
            c_row[j + 3] = s3;
            c_row[j + 4] = s4;
            c_row[j + 5] = s5;
            c_row[j + 6] = s6;
            c_row[j + 7] = s7;
        }

        for (; j < n; ++j) {
            const float* bt_row = bt + static_cast<std::size_t>(j) * n;

            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a_row[k] * bt_row[k];
            }

            c_row[j] = sum;
        }
    }
}

void naive_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    naive_matmul(args.C, args.A, args.B, args.n);
}

void stu_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    stu_matmul(args.C, args.A, args.B, args.n);
}

bool matmul_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<matmul_args*>(stu_ctx);
    auto& ref_args = *static_cast<matmul_args*>(ref_ctx);

    if (stu_args.C.size() != ref_args.C.size()) {
        debug_log("\tDEBUG: matmul size mismatch: stu={} ref={}\n",
                  stu_args.C.size(),
                  ref_args.C.size());
        return false;
    }

    const double eps = ref_args.epsilon;
    const int n = ref_args.n;
    double max_rel = 0.0;
    size_t worst_idx = 0;

    for (size_t i = 0; i < ref_args.C.size(); ++i) {
        const double r = static_cast<double>(ref_args.C[i]);
        const double s = static_cast<double>(stu_args.C[i]);
        const double diff = std::abs(s - r);
        const double rel = (std::abs(r) > 1e-9) ? diff / std::abs(r) : diff;

        if (rel > max_rel) {
            max_rel = rel;
            worst_idx = i;
        }

        if (rel > eps) {
            const size_t row = (n > 0) ? (i / static_cast<size_t>(n)) : 0;
            const size_t col = (n > 0) ? (i % static_cast<size_t>(n)) : 0;
            debug_log("\tDEBUG: matmul fail at index {} (row={}, col={}): ref={} stu={} rel={} eps={}\n",
                      i,
                      row,
                      col,
                      ref_args.C[i],
                      stu_args.C[i],
                      rel,
                      eps);
            return false;
        }
    }

    debug_log("\tDEBUG: matmul_check passed. max_rel={} at index {}\n",
              max_rel,
              worst_idx);
    return true;
}
