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
    constexpr float inv9 = 1.0f / 9.0f;

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
        const std::size_t row_m = (y - 1) * W;
        const std::size_t row_0 = y * W;
        const std::size_t row_p = (y + 1) * W;

        const float* __restrict__ am = a + row_m;
        const float* __restrict__ a0 = a + row_0;
        const float* __restrict__ ap = a + row_p;

        const float* __restrict__ bm = b + row_m;
        const float* __restrict__ b0 = b + row_0;
        const float* __restrict__ bp = b + row_p;

        const float* __restrict__ cm = c + row_m;
        const float* __restrict__ c0 = c + row_0;
        const float* __restrict__ cp = c + row_p;

        const float* __restrict__ dm = d + row_m;
        const float* __restrict__ d0 = d + row_0;
        const float* __restrict__ dp = d + row_p;

        const float* __restrict__ em = e + row_m;
        const float* __restrict__ e0 = e + row_0;
        const float* __restrict__ ep = e + row_p;

        const float* __restrict__ fm = f + row_m;
        const float* __restrict__ f0 = f + row_0;
        const float* __restrict__ fp = f + row_p;

        const float* __restrict__ gm = g + row_m;
        const float* __restrict__ gp = g + row_p;

        const float* __restrict__ hm = h + row_m;
        const float* __restrict__ hp = h + row_p;

        const float* __restrict__ im = i_ch + row_m;
        const float* __restrict__ ip = i_ch + row_p;

        for (std::size_t x = 1; x + 1 < W; ++x) {
            const std::size_t xm1 = x - 1;
            const std::size_t xp1 = x + 1;

            const double sum_a =
                static_cast<double>(am[xm1]) + am[x] + am[xp1] +
                static_cast<double>(a0[xm1]) + a0[x] + a0[xp1] +
                static_cast<double>(ap[xm1]) + ap[x] + ap[xp1];

            const double sum_b =
                static_cast<double>(bm[xm1]) + bm[x] + bm[xp1] +
                static_cast<double>(b0[xm1]) + b0[x] + b0[xp1] +
                static_cast<double>(bp[xm1]) + bp[x] + bp[xp1];

            const double sum_c =
                static_cast<double>(cm[xm1]) + cm[x] + cm[xp1] +
                static_cast<double>(c0[xm1]) + c0[x] + c0[xp1] +
                static_cast<double>(cp[xm1]) + cp[x] + cp[xp1];

            const float avg_a = static_cast<float>(sum_a * inv9);
            const float avg_b = static_cast<float>(sum_b * inv9);
            const float avg_c = static_cast<float>(sum_c * inv9);

            const float p1 = avg_a * avg_b + avg_c;

            const float sobel_dx =
                -dm[xm1] + dm[xp1]
                -2.0f * d0[xm1] + 2.0f * d0[xp1]
                -dp[xm1] + dp[xp1];

            const float sobel_ex =
                -em[xm1] + em[xp1]
                -2.0f * e0[xm1] + 2.0f * e0[xp1]
                -ep[xm1] + ep[xp1];

            const float sobel_fx =
                -fm[xm1] + fm[xp1]
                -2.0f * f0[xm1] + 2.0f * f0[xp1]
                -fp[xm1] + fp[xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -gm[xm1] - 2.0f * gm[x] - gm[xp1]
                + gp[xm1] + 2.0f * gp[x] + gp[xp1];

            const float sobel_hy =
                -hm[xm1] - 2.0f * hm[x] - hm[xp1]
                + hp[xm1] + 2.0f * hp[x] + hp[xp1];

            const float sobel_iy =
                -im[xm1] - 2.0f * im[x] - im[xp1]
                + ip[xm1] + 2.0f * ip[x] + ip[xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += static_cast<double>(p1 + p2 + p3);
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
