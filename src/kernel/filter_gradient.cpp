#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

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
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr double inv9 = 1.0 / 9.0;

    const float* __restrict__ a = data.a.data();
    const float* __restrict__ b = data.b.data();
    const float* __restrict__ c = data.c.data();
    const float* __restrict__ d = data.d.data();
    const float* __restrict__ e = data.e.data();
    const float* __restrict__ f = data.f.data();
    const float* __restrict__ g = data.g.data();
    const float* __restrict__ h = data.h.data();
    const float* __restrict__ i_ch = data.i.data();

    double total = 0.0;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        const std::size_t ym1 = (y - 1) * W;
        const std::size_t y0  = y * W;
        const std::size_t yp1 = (y + 1) * W;

        auto colsum_a = [&](std::size_t x) -> double {
            return static_cast<double>(a[ym1 + x]) +
                   static_cast<double>(a[y0  + x]) +
                   static_cast<double>(a[yp1 + x]);
        };

        auto colsum_b = [&](std::size_t x) -> double {
            return static_cast<double>(b[ym1 + x]) +
                   static_cast<double>(b[y0  + x]) +
                   static_cast<double>(b[yp1 + x]);
        };

        auto colsum_c = [&](std::size_t x) -> double {
            return static_cast<double>(c[ym1 + x]) +
                   static_cast<double>(c[y0  + x]) +
                   static_cast<double>(c[yp1 + x]);
        };

        double ca0 = colsum_a(0);
        double ca1 = colsum_a(1);
        double ca2 = colsum_a(2);

        double cb0 = colsum_b(0);
        double cb1 = colsum_b(1);
        double cb2 = colsum_b(2);

        double cc0 = colsum_c(0);
        double cc1 = colsum_c(1);
        double cc2 = colsum_c(2);

        for (std::size_t x = 1; x + 1 < W; ++x) {
            const std::size_t xm1 = x - 1;
            const std::size_t xp1 = x + 1;

            const float avg_a = static_cast<float>((ca0 + ca1 + ca2) * inv9);
            const float avg_b = static_cast<float>((cb0 + cb1 + cb2) * inv9);
            const float avg_c = static_cast<float>((cc0 + cc1 + cc2) * inv9);

            const float p1 = avg_a * avg_b + avg_c;

            const float sobel_dx =
                -d[ym1 + xm1] + d[ym1 + xp1]
                -2.0f * d[y0 + xm1] + 2.0f * d[y0 + xp1]
                -d[yp1 + xm1] + d[yp1 + xp1];

            const float sobel_ex =
                -e[ym1 + xm1] + e[ym1 + xp1]
                -2.0f * e[y0 + xm1] + 2.0f * e[y0 + xp1]
                -e[yp1 + xm1] + e[yp1 + xp1];

            const float sobel_fx =
                -f[ym1 + xm1] + f[ym1 + xp1]
                -2.0f * f[y0 + xm1] + 2.0f * f[y0 + xp1]
                -f[yp1 + xm1] + f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -g[ym1 + xm1] - 2.0f * g[ym1 + x] - g[ym1 + xp1]
                + g[yp1 + xm1] + 2.0f * g[yp1 + x] + g[yp1 + xp1];

            const float sobel_hy =
                -h[ym1 + xm1] - 2.0f * h[ym1 + x] - h[ym1 + xp1]
                + h[yp1 + xm1] + 2.0f * h[yp1 + x] + h[yp1 + xp1];

            const float sobel_iy =
                -i_ch[ym1 + xm1] - 2.0f * i_ch[ym1 + x] - i_ch[ym1 + xp1]
                + i_ch[yp1 + xm1] + 2.0f * i_ch[yp1 + x] + i_ch[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += static_cast<double>(p1 + p2 + p3);

            if (x + 2 < W) {
                ca0 = ca1;
                ca1 = ca2;
                ca2 = colsum_a(x + 2);

                cb0 = cb1;
                cb1 = cb2;
                cb2 = colsum_b(x + 2);

                cc0 = cc1;
                cc1 = cc2;
                cc2 = colsum_c(x + 2);
            }
        }
    }

    out = static_cast<float>(total);
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
