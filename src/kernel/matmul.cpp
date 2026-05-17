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
    // TODO: Implement your version, and call it in stu_matmul_wrapper

    std::fill(C.begin(), C.end(), 0.0f);

    const float* __restrict__ a = A.data();
    const float* __restrict__ b = B.data();
    float* __restrict__ c = C.data();

    constexpr int BS = 32;

    for (int ii = 0; ii < n; ii += BS) {
        const int i_end = std::min(ii + BS, n);

        for (int kk = 0; kk < n; kk += BS) {
            const int k_end = std::min(kk + BS, n);

            for (int jj = 0; jj < n; jj += BS) {
                const int j_end = std::min(jj + BS, n);

                for (int i = ii; i < i_end; ++i) {
                    const int i_base = i * n;

                    for (int k = kk; k < k_end; ++k) {
                        const float aik = a[i_base + k];
                        const int k_base = k * n;

                        int j = jj;

                        for (; j + 7 < j_end; j += 8) {
                            c[i_base + j + 0] += aik * b[k_base + j + 0];
                            c[i_base + j + 1] += aik * b[k_base + j + 1];
                            c[i_base + j + 2] += aik * b[k_base + j + 2];
                            c[i_base + j + 3] += aik * b[k_base + j + 3];
                            c[i_base + j + 4] += aik * b[k_base + j + 4];
                            c[i_base + j + 5] += aik * b[k_base + j + 5];
                            c[i_base + j + 6] += aik * b[k_base + j + 6];
                            c[i_base + j + 7] += aik * b[k_base + j + 7];
                        }

                        for (; j < j_end; ++j) {
                            c[i_base + j] += aik * b[k_base + j];
                        }
                    }
                }
            }
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
