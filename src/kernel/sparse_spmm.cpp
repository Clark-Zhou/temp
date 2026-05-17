#include "sparse_spmm.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>

/*
Sparse matrix LHS: csr.row x csr.col
Transposed dense RHS: dense_cols x csr.cols

Input
    csr: [csr.row, csr.col]
    dense_t: [dense_cols, csr.col]
Output
    out: [csr.row, dense_cols]
*/

static unsigned int sparse_runtime_salt() {
    static const unsigned int salt = [] {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd()};
        std::array<unsigned int, 1> values{};
        seq.generate(values.begin(), values.end());
        return values[0];
    }();
    return salt;
}

static CSRMatrix build_sparse_matrix(int block_row_count, int block_col_count,
                                     const std::vector<int> &diagonal_offsets,
                                     unsigned int seed) {
    if (block_row_count <= 0 || block_col_count <= 0) {
        throw std::invalid_argument(
            "initialize_spmm: block counts must be positive.");
    }

    const int min_offset = -(block_row_count - 1);
    const int max_offset = block_col_count - 1;

    std::vector<int> offsets;
    if (diagonal_offsets.empty()) {
        constexpr int kDefaultOffsets[] = {-456, -123, 0, 137, 246};
        for (int offset : kDefaultOffsets) {
            if (offset >= min_offset && offset <= max_offset) {
                offsets.push_back(offset);
            }
        }
        if (offsets.empty()) {
            offsets.push_back(0);
        }
    } else {
        offsets = diagonal_offsets;
        for (size_t i = 1; i < offsets.size(); ++i) {
            if (offsets[i - 1] >= offsets[i]) {
                throw std::invalid_argument(
                    "initialize_spmm: diagonal_offsets must be strictly "
                    "increasing.");
            }
        }
        for (int offset : offsets) {
            if (offset < min_offset || offset > max_offset) {
                throw std::invalid_argument(
                    "initialize_spmm: diagonal_offsets contain an out-of-range "
                    "value.");
            }
        }
    }

    CSRMatrix csr;
    csr.rows = block_row_count * 4;
    csr.cols = block_col_count * 4;

    std::vector<std::vector<int>> row_cols(csr.rows);
    std::vector<std::vector<float>> row_vals(csr.rows);
    const size_t n_offsets = offsets.size();
    for (int r = 0; r < csr.rows; ++r) {
        row_cols[r].reserve(n_offsets * 4);
        row_vals[r].reserve(n_offsets * 4);
    }

    // Keep the diagonal pattern fixed while varying the stored values each run.
    std::seed_seq value_seed{seed, sparse_runtime_salt(), 0x85ebca6bu};
    std::mt19937 rng(value_seed);
    std::uniform_int_distribution<int> value_dist(-10, 10);

    for (int br = 0; br < block_row_count; ++br) {
        for (size_t od = 0; od < n_offsets; ++od) {
            const int bc = br + offsets[od];
            if (bc < 0 || bc >= block_col_count)
                continue;

            for (int lr = 0; lr < 4; ++lr) {
                const int row = br * 4 + lr;
                std::vector<int> &cols = row_cols[row];
                std::vector<float> &vals = row_vals[row];

                for (int lc = 0; lc < 4; ++lc) {
                    const int v = value_dist(rng);
                    if (v == 0)
                        continue; // CSR should not store explicit zeros.

                    cols.push_back(bc * 4 + lc);
                    vals.push_back(v);
                }
            }
        }
    }

    csr.row_ptr.assign(csr.rows + 1, 0);
    for (int r = 0; r < csr.rows; ++r) {
        const int row_nnz = row_cols[r].size();
        csr.row_ptr[r + 1] = csr.row_ptr[r] + row_nnz;
    }

    const int total_nnz = csr.row_ptr.back();
    csr.col_idx.resize(total_nnz);
    csr.values.resize(total_nnz);

    for (int r = 0; r < csr.rows; ++r) {
        int out = csr.row_ptr[r];
        for (size_t k = 0; k < row_cols[r].size(); ++k) {
            csr.col_idx[out] = row_cols[r][k];
            csr.values[out] = row_vals[r][k];
            ++out;
        }
    }

    return csr;
}

void initialize_spmm(sparse_spmm_args &args, int block_row_count,
                          int block_col_count, int dense_cols,
                          const std::vector<int> &diagonal_offsets,
                          unsigned int seed) {
    args.csr = build_sparse_matrix(block_row_count,
                                   block_col_count,
                                   diagonal_offsets,
                                   seed);
    // print_dense_matrix(args.csr);

    if (dense_cols <= 0) {
        // Default to square dense RHS when possible.
        dense_cols = args.csr.cols;
    }

    const size_t dense_cols_sz = dense_cols;
    const size_t csr_cols_sz = args.csr.cols;
    const size_t csr_rows_sz = args.csr.rows;
    args.dense_t.resize(dense_cols_sz * csr_cols_sz);
    args.out.resize(csr_rows_sz * dense_cols_sz);
    args.epsilon = 1e-3;

    std::seed_seq dense_seed{seed, sparse_runtime_salt(), 0x9e3779b9u};
    std::mt19937 rng(dense_seed);
    std::uniform_real_distribution<float> dense_dist(-1.0f, 1.0f);
    for (size_t i = 0; i < args.dense_t.size(); ++i) {
        args.dense_t[i] = dense_dist(rng);
    }

    // Pre-touch outputs and inputs once to reduce first-call page faults.
    std::fill(args.out.begin(), args.out.end(), 0.0f);
    volatile float touch = 0.0f;
    for (size_t i = 0; i < args.dense_t.size(); ++i)
        touch = touch + args.dense_t[i];
    for (size_t i = 0; i < args.out.size(); ++i)
        touch = touch + args.out[i];
    for (size_t i = 0; i < args.csr.values.size(); ++i)
        touch = touch + args.csr.values[i];
    (void)touch;
}


void csr_spmm(const CSRMatrix &csr, const std::vector<float> &dense_t,
              std::vector<float> &out) {
    if (!validate_csr(csr)) {
        throw std::invalid_argument("csr_spmm: invalid CSR matrix.");
    }
    const size_t rows = csr.rows;
    const size_t cols = csr.cols;
    if (rows == 0 || cols == 0) {
        if (!dense_t.empty() || !out.empty()) {
            throw std::invalid_argument(
                "csr_spmm: non-empty dense buffers for empty CSR shape.");
        }
        return;
    }
    if (dense_t.size() % cols != 0) {
        throw std::invalid_argument(
            "csr_spmm: dense_t.size() must be a multiple of csr.cols.");
    }
    const size_t dense_cols = dense_t.size() / cols;
    if (dense_cols == 0) {
        throw std::invalid_argument("csr_spmm: dense_cols must be positive.");
    }
    if (out.size() != rows * dense_cols) {
        throw std::invalid_argument("csr_spmm: out size mismatch.");
    }

    for (int r = 0; r < csr.rows; ++r) {
        float *out_row = &out[r * dense_cols];
        for (size_t n = 0; n < dense_cols; ++n) {
            const float *bt_row = &dense_t[n * cols];
            float acc = 0.0f;
            for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
                acc += csr.values[p] * bt_row[csr.col_idx[p]];
            }
            out_row[n] = acc;
        }
    }
}

void naive_sparse_spmm_wrapper(void *ctx) {
    auto &args = *static_cast<sparse_spmm_args *>(ctx);
    csr_spmm(args.csr, args.dense_t, args.out);
}

// TODO: Implement your version (e.g. stu_csr_spmm), and call it in stu_sparse_spmm_wrapper

void stu_csr_spmm(const CSRMatrix &csr, const std::vector<float> &dense_t,
                  std::vector<float> &out) {
    if (!validate_csr(csr)) {
        throw std::invalid_argument("stu_csr_spmm: invalid CSR matrix.");
    }

    const int rows = csr.rows;
    const int cols = csr.cols;
    if (rows == 0 || cols == 0) {
        return;
    }
    if (dense_t.size() % static_cast<std::size_t>(cols) != 0) {
        throw std::invalid_argument(
            "stu_csr_spmm: dense_t.size() must be a multiple of csr.cols.");
    }

    const int dense_cols = static_cast<int>(dense_t.size() / cols);
    if (out.size() != static_cast<std::size_t>(rows) * dense_cols) {
        throw std::invalid_argument("stu_csr_spmm: out size mismatch.");
    }

    const int *row_ptr = csr.row_ptr.data();
    const int *col_idx = csr.col_idx.data();
    const float *values = csr.values.data();
    const float *dense = dense_t.data();
    float *output = out.data();

    int r = 0;
    for (; r + 3 < rows; r += 4) {
        const int row0_begin = row_ptr[r];
        const int row0_end = row_ptr[r + 1];
        const int row1_begin = row_ptr[r + 1];
        const int row1_end = row_ptr[r + 2];
        const int row2_begin = row_ptr[r + 2];
        const int row2_end = row_ptr[r + 3];
        const int row3_begin = row_ptr[r + 3];
        const int row3_end = row_ptr[r + 4];

        float *out0 = output + static_cast<std::size_t>(r) * dense_cols;
        float *out1 = out0 + dense_cols;
        float *out2 = out1 + dense_cols;
        float *out3 = out2 + dense_cols;

        for (int n = 0; n < dense_cols; ++n) {
            const float *dense_row = dense + static_cast<std::size_t>(n) * cols;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            for (int p = row0_begin; p < row0_end; ++p) {
                acc0 += values[p] * dense_row[col_idx[p]];
            }
            for (int p = row1_begin; p < row1_end; ++p) {
                acc1 += values[p] * dense_row[col_idx[p]];
            }
            for (int p = row2_begin; p < row2_end; ++p) {
                acc2 += values[p] * dense_row[col_idx[p]];
            }
            for (int p = row3_begin; p < row3_end; ++p) {
                acc3 += values[p] * dense_row[col_idx[p]];
            }

            out0[n] = acc0;
            out1[n] = acc1;
            out2[n] = acc2;
            out3[n] = acc3;
        }
    }

    for (; r < rows; ++r) {
        float *out_row = output + static_cast<std::size_t>(r) * dense_cols;
        const int row_begin = row_ptr[r];
        const int row_end = row_ptr[r + 1];

        for (int n = 0; n < dense_cols; ++n) {
            const float *dense_row = dense + static_cast<std::size_t>(n) * cols;
            float acc = 0.0f;
            for (int p = row_begin; p < row_end; ++p) {
                acc += values[p] * dense_row[col_idx[p]];
            }
            out_row[n] = acc;
        }
    }
}

void stu_sparse_spmm_wrapper(void *ctx) {
    auto &args = *static_cast<sparse_spmm_args *>(ctx);
    const CSRMatrix& csr = args.csr;

    if (!validate_csr(csr) || csr.rows <= 0 || csr.cols <= 0) {
        stu_csr_spmm(csr, args.dense_t, args.out);
        return;
    }

    const int rows = csr.rows;
    const int cols = csr.cols;
    const int dense_cols = static_cast<int>(args.dense_t.size() / cols);
    if (dense_cols <= 0 ||
        args.out.size() != static_cast<std::size_t>(rows) * dense_cols) {
        stu_csr_spmm(csr, args.dense_t, args.out);
        return;
    }

    constexpr std::size_t kThreadCount = 8;
    const int* row_ptr = csr.row_ptr.data();
    const int* col_idx = csr.col_idx.data();
    const float* values = csr.values.data();
    const float* dense = args.dense_t.data();
    float* output = args.out.data();

    auto compute_rows = [&](int row_begin, int row_end) {
        for (int r = row_begin; r < row_end; ++r) {
            float* out_row = output + static_cast<std::size_t>(r) * dense_cols;
            const int begin = row_ptr[r];
            const int end = row_ptr[r + 1];

            int n = 0;
            for (; n + 3 < dense_cols; n += 4) {
                const float* d0 = dense + static_cast<std::size_t>(n) * cols;
                const float* d1 = d0 + cols;
                const float* d2 = d1 + cols;
                const float* d3 = d2 + cols;
                float s0 = 0.0f;
                float s1 = 0.0f;
                float s2 = 0.0f;
                float s3 = 0.0f;

                for (int p = begin; p < end; ++p) {
                    const int col = col_idx[p];
                    const float v = values[p];
                    s0 += v * d0[col];
                    s1 += v * d1[col];
                    s2 += v * d2[col];
                    s3 += v * d3[col];
                }
                out_row[n] = s0;
                out_row[n + 1] = s1;
                out_row[n + 2] = s2;
                out_row[n + 3] = s3;
            }

            for (; n < dense_cols; ++n) {
                const float* dense_row = dense + static_cast<std::size_t>(n) * cols;
                float acc = 0.0f;
                for (int p = begin; p < end; ++p) {
                    acc += values[p] * dense_row[col_idx[p]];
                }
                out_row[n] = acc;
            }
        }
    };

    if (rows >= static_cast<int>(kThreadCount * 32)) {
        std::array<std::thread, kThreadCount - 1> workers;
        for (std::size_t t = 0; t + 1 < kThreadCount; ++t) {
            const int begin = static_cast<int>((static_cast<std::size_t>(rows) * t) / kThreadCount);
            const int end = static_cast<int>((static_cast<std::size_t>(rows) * (t + 1)) / kThreadCount);
            workers[t] = std::thread(compute_rows, begin, end);
        }
        const int main_begin = static_cast<int>((static_cast<std::size_t>(rows) * (kThreadCount - 1)) / kThreadCount);
        compute_rows(main_begin, rows);
        for (auto& worker : workers) {
            worker.join();
        }
    } else {
        compute_rows(0, rows);
    }
}

bool sparse_spmm_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<sparse_spmm_args *>(stu_ctx);
    auto &ref_args = *static_cast<sparse_spmm_args *>(ref_ctx);
    const double eps = ref_args.epsilon;
    if (stu_args.out.size() != ref_args.out.size())
        return false;

    const double atol = 2e-6;
    for (size_t i = 0; i < ref_args.out.size(); ++i) {
        const double r = static_cast<double>(ref_args.out[i]);
        const double s = static_cast<double>(stu_args.out[i]);
        const double err = std::abs(r - s);
        if (err > (atol + eps * std::abs(r))) {
            debug_log("\tDEBUG: sparse_spmm mismatch at {}: ref={} stu={} err={} thr={}\n",
                      i,
                      r,
                      s,
                      err,
                      (atol + eps * std::abs(r)));
            return false;
        }
    }
    return true;
}
