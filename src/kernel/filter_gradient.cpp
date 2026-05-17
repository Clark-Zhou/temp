#include "filter_gradient.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {

            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                -data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                -data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                -data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1]
                + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1]
                + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1]
                + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

void stu_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    // TODO: You may need to add a function to convert data structure (not
    // included in time measurement), then implement your version in
    // stu_filter_gradient, whch is called by stu_filter_gradient_wrapper.

    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    constexpr std::size_t kThreadCount = 4;

    if (W < 3 || H < 3) {
        out = 0.0f;
        return;
    }

    const float *a = data.a.data();
    const float *b = data.b.data();
    const float *c = data.c.data();
    const float *d = data.d.data();
    const float *e = data.e.data();
    const float *f = data.f.data();
    const float *g = data.g.data();
    const float *h = data.h.data();
    const float *i_channel = data.i.data();

    auto compute_rows = [&](std::size_t y_begin, std::size_t y_end) {
        double local_total = 0.0;

        for (std::size_t y = y_begin; y < y_end; ++y) {
            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0 = y * W;
            const std::size_t yp1 = (y + 1) * W;

            double a_col0 = static_cast<double>(a[ym1]) + a[y0] + a[yp1];
            double a_col1 = static_cast<double>(a[ym1 + 1]) + a[y0 + 1] +
                            a[yp1 + 1];
            double a_col2 = static_cast<double>(a[ym1 + 2]) + a[y0 + 2] +
                            a[yp1 + 2];
            double b_col0 = static_cast<double>(b[ym1]) + b[y0] + b[yp1];
            double b_col1 = static_cast<double>(b[ym1 + 1]) + b[y0 + 1] +
                            b[yp1 + 1];
            double b_col2 = static_cast<double>(b[ym1 + 2]) + b[y0 + 2] +
                            b[yp1 + 2];
            double c_col0 = static_cast<double>(c[ym1]) + c[y0] + c[yp1];
            double c_col1 = static_cast<double>(c[ym1 + 1]) + c[y0 + 1] +
                            c[yp1 + 1];
            double c_col2 = static_cast<double>(c[ym1 + 2]) + c[y0 + 2] +
                            c[yp1 + 2];

            for (std::size_t x = 1; x + 1 < W; ++x) {
                const std::size_t xm1 = x - 1;
                const std::size_t x0 = x;
                const std::size_t xp1 = x + 1;

                const std::size_t p00 = ym1 + xm1;
                const std::size_t p01 = ym1 + x0;
                const std::size_t p02 = ym1 + xp1;
                const std::size_t p10 = y0 + xm1;
                const std::size_t p12 = y0 + xp1;
                const std::size_t p20 = yp1 + xm1;
                const std::size_t p21 = yp1 + x0;
                const std::size_t p22 = yp1 + xp1;

                const double sum_a = a_col0 + a_col1 + a_col2;
                const double sum_b = b_col0 + b_col1 + b_col2;
                const double sum_c = c_col0 + c_col1 + c_col2;
                const float avg_a = sum_a * inv9;
                const float avg_b = sum_b * inv9;
                const float avg_c = sum_c * inv9;
                const float p1 = avg_a * avg_b + avg_c;

                const float sobel_dx = -d[p00] + d[p02] - 2.0f * d[p10] +
                                       2.0f * d[p12] - d[p20] + d[p22];
                const float sobel_ex = -e[p00] + e[p02] - 2.0f * e[p10] +
                                       2.0f * e[p12] - e[p20] + e[p22];
                const float sobel_fx = -f[p00] + f[p02] - 2.0f * f[p10] +
                                       2.0f * f[p12] - f[p20] + f[p22];
                const float p2 = sobel_dx * sobel_ex + sobel_fx;

                const float sobel_gy = -g[p00] - 2.0f * g[p01] - g[p02] +
                                       g[p20] + 2.0f * g[p21] + g[p22];
                const float sobel_hy = -h[p00] - 2.0f * h[p01] - h[p02] +
                                       h[p20] + 2.0f * h[p21] + h[p22];
                const float sobel_iy = -i_channel[p00] - 2.0f * i_channel[p01] -
                                       i_channel[p02] + i_channel[p20] +
                                       2.0f * i_channel[p21] + i_channel[p22];
                const float p3 = sobel_gy * sobel_hy + sobel_iy;

                local_total += p1 + p2 + p3;

                if (x + 2 < W) {
                    const std::size_t next = x + 2;
                    a_col0 = a_col1;
                    a_col1 = a_col2;
                    a_col2 = static_cast<double>(a[ym1 + next]) + a[y0 + next] +
                             a[yp1 + next];
                    b_col0 = b_col1;
                    b_col1 = b_col2;
                    b_col2 = static_cast<double>(b[ym1 + next]) + b[y0 + next] +
                             b[yp1 + next];
                    c_col0 = c_col1;
                    c_col1 = c_col2;
                    c_col2 = static_cast<double>(c[ym1 + next]) + c[y0 + next] +
                             c[yp1 + next];
                }
            }
        }

        return local_total;
    };

    const std::size_t interior_rows = H - 2;
    if (interior_rows >= kThreadCount * 64) {
        std::array<std::thread, kThreadCount - 1> workers;
        std::array<double, kThreadCount> partials{};

        for (std::size_t t = 0; t + 1 < kThreadCount; ++t) {
            const std::size_t y_begin = 1 + (interior_rows * t) / kThreadCount;
            const std::size_t y_end = 1 + (interior_rows * (t + 1)) / kThreadCount;
            workers[t] = std::thread([&, t, y_begin, y_end] {
                partials[t] = compute_rows(y_begin, y_end);
            });
        }

        const std::size_t main_begin = 1 + (interior_rows * (kThreadCount - 1)) /
                                           kThreadCount;
        partials[kThreadCount - 1] = compute_rows(main_begin, H - 1);

        for (auto& worker : workers) {
            worker.join();
        }

        double total = 0.0;
        for (double part : partials) {
            total += part;
        }
        out = total;
    } else {
        out = compute_rows(1, H - 1);
    }
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
