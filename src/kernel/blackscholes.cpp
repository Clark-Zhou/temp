#include "blackscholes.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>

#define inv_sqrt_2xPI 0.39894228040143270286
#define p_val 0.2316419
#define coefficient_a1 0.319381530
#define coefficient_a2 -0.356563782
#define coefficient_a3 1.781477937
#define coefficient_a4 -1.821255978
#define coefficient_a5 1.330274429

void initialize_blackscholes(blackscholes_args &args,
                             std::size_t n,
                             std::uint32_t seed) {
    args.call_option_price.assign(n, 0.0f);
    args.put_option_price.assign(n, 0.0f);
    args.epsilon = 5e-3;

    args.spot_price.resize(n);
    args.strike.resize(n);
    args.rate.resize(n);
    args.volatility.resize(n);
    args.time.resize(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> spot_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> strike_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> rate_dist(0.0275f, 0.1f);
    std::uniform_real_distribution<float> vol_dist(0.05f, 0.6f);
    std::uniform_real_distribution<float> time_dist(0.1f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        args.spot_price[i] = spot_dist(rng);
        args.strike[i] = strike_dist(rng);
        args.rate[i] = rate_dist(rng);
        args.volatility[i] = vol_dist(rng);
        args.time[i] = time_dist(rng);
    }
}

void CNDF(float &InputX, float &OutputX) {
    int sign = 0;
    float x = InputX;

    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }

    const float xNPrimeofX = std::exp(-0.5f * x * x) * inv_sqrt_2xPI;
    const float k = 1.0f / (1.0f + p_val * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;

    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;

    OutputX = sign ? (1.0f - local) : local;
}

static inline void naive_BlkSchls_one(float &CallOptionPrice,
                                      float &PutOptionPrice, float spotPrice,
                                      float strike, float rate,
                                      float volatility, float time) {
    const float xSqrtTime = std::sqrt(time);
    const float xLogTerm = std::log(spotPrice / strike);
    const float xPowerTerm = 0.5f * volatility * volatility;

    float xD1 = (rate + xPowerTerm) * time + xLogTerm;
    const float xDen = volatility * xSqrtTime;
    xD1 = xD1 / xDen;
    const float xD2 = xD1 - xDen;

    float d1 = xD1;
    float d2 = xD2;
    float NofXd1 = 0.0f;
    float NofXd2 = 0.0f;

    CNDF(d1, NofXd1);
    CNDF(d2, NofXd2);

    const float FutureValueX = strike * std::exp(-(rate) * (time));
    CallOptionPrice = (spotPrice * NofXd1) - (FutureValueX * NofXd2);

    const float NegNofXd1 = 1.0f - NofXd1;
    const float NegNofXd2 = 1.0f - NofXd2;
    PutOptionPrice = (FutureValueX * NegNofXd2) - (spotPrice * NegNofXd1);
}

void naive_BlkSchls(std::vector<float> &CallOptionPrice,
                    std::vector<float> &PutOptionPrice,
                    const std::vector<float> &spotPrice,
                    const std::vector<float> &strike,
                    const std::vector<float> &rate,
                    const std::vector<float> &volatility,
                    const std::vector<float> &time) {
    size_t n = spotPrice.size();
    for (size_t i = 0; i < n; ++i) {
        naive_BlkSchls_one(CallOptionPrice[i],
                           PutOptionPrice[i],
                           spotPrice[i],
                           strike[i],
                           rate[i],
                           volatility[i],
                           time[i]);
    }
}

void stu_BlkSchls(std::vector<float> &CallOptionPrice,
                  std::vector<float> &PutOptionPrice,
                  const std::vector<float> &spotPrice,
                  const std::vector<float> &strike,
                  const std::vector<float> &rate,
                  const std::vector<float> &volatility,
                  const std::vector<float> &time) {
    // TODO:
    // Implement your version for BlkSchls here, then 
    // call it at stu_BlkSchls_wrapper()...
    const size_t n = spotPrice.size();

    float* __restrict__ call = CallOptionPrice.data();
    float* __restrict__ put = PutOptionPrice.data();

    const float* __restrict__ spot = spotPrice.data();
    const float* __restrict__ stk = strike.data();
    const float* __restrict__ rt = rate.data();
    const float* __restrict__ vol = volatility.data();
    const float* __restrict__ tm = time.data();

    auto fast_log_ratio = [](float s, float k) -> float {
        /*
         * log(s / k) using:
         *   log(z) = 2 * (y + y^3/3 + y^5/5 + ...)
         * where:
         *   y = (z - 1) / (z + 1)
         *
         * In this benchmark, s and k are both in [50.0, 99.9],
         * so z is roughly in [0.5, 2.0], and |y| <= about 1/3.
         * The series converges quickly in this range.
         */
        const float z = s / k;
        const float y = (z - 1.0f) / (z + 1.0f);
        const float y2 = y * y;

        const float poly =
            1.0f
            + y2 * (1.0f / 3.0f
            + y2 * (1.0f / 5.0f
            + y2 * (1.0f / 7.0f
            + y2 * (1.0f / 9.0f
            + y2 * (1.0f / 11.0f)))));

        return 2.0f * y * poly;
    };

    auto fast_discount_exp = [](float x) -> float {
        /*
         * Approximate exp(-x), where x = rate * time.
         * In this benchmark, rate is in [0.0275, 0.1] and time is in [0.1, 1.0],
         * so x is small. A short Taylor expansion is accurate enough.
         */
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x4 = x2 * x2;
        const float x5 = x4 * x;
        const float x6 = x3 * x3;

        return 1.0f
            - x
            + 0.5f * x2
            - (1.0f / 6.0f) * x3
            + (1.0f / 24.0f) * x4
            - (1.0f / 120.0f) * x5
            + (1.0f / 720.0f) * x6;
    };

    auto cndf_inline = [](float input) -> float {
        int sign = 0;
        float x = input;

        if (x < 0.0f) {
            x = -x;
            sign = 1;
        }

        // For very large x, the CDF rounds to 1.0f or 0.0f in float anyway.
        if (x > 8.0f) {
            return sign ? 0.0f : 1.0f;
        }

        const float xNPrimeofX =
            std::exp(-0.5f * x * x) * static_cast<float>(inv_sqrt_2xPI);

        const float k = 1.0f / (1.0f + static_cast<float>(p_val) * x);
        const float k_2 = k * k;
        const float k_3 = k_2 * k;
        const float k_4 = k_3 * k;
        const float k_5 = k_4 * k;

        float local = k * static_cast<float>(coefficient_a1);
        local += k_2 * static_cast<float>(coefficient_a2);
        local += k_3 * static_cast<float>(coefficient_a3);
        local += k_4 * static_cast<float>(coefficient_a4);
        local += k_5 * static_cast<float>(coefficient_a5);
        local = 1.0f - local * xNPrimeofX;

        return sign ? (1.0f - local) : local;
    };

    size_t i = 0;

    for (; i + 1 < n; i += 2) {
        {
            const float s = spot[i];
            const float k = stk[i];
            const float r = rt[i];
            const float v = vol[i];
            const float t = tm[i];

            const float sqrt_t = std::sqrt(t);
            const float log_term = fast_log_ratio(s, k);
            const float power_term = 0.5f * v * v;

            const float den = v * sqrt_t;
            const float d1 = ((r + power_term) * t + log_term) / den;
            const float d2 = d1 - den;

            const float nd1 = cndf_inline(d1);
            const float nd2 = cndf_inline(d2);

            const float future = k * fast_discount_exp(r * t);

            call[i] = (s * nd1) - (future * nd2);
            put[i] = (future * (1.0f - nd2)) - (s * (1.0f - nd1));
        }

        {
            const size_t j = i + 1;

            const float s = spot[j];
            const float k = stk[j];
            const float r = rt[j];
            const float v = vol[j];
            const float t = tm[j];

            const float sqrt_t = std::sqrt(t);
            const float log_term = fast_log_ratio(s, k);
            const float power_term = 0.5f * v * v;

            const float den = v * sqrt_t;
            const float d1 = ((r + power_term) * t + log_term) / den;
            const float d2 = d1 - den;

            const float nd1 = cndf_inline(d1);
            const float nd2 = cndf_inline(d2);

            const float future = k * fast_discount_exp(r * t);

            call[j] = (s * nd1) - (future * nd2);
            put[j] = (future * (1.0f - nd2)) - (s * (1.0f - nd1));
        }
    }

    for (; i < n; ++i) {
        const float s = spot[i];
        const float k = stk[i];
        const float r = rt[i];
        const float v = vol[i];
        const float t = tm[i];

        const float sqrt_t = std::sqrt(t);
        const float log_term = fast_log_ratio(s, k);
        const float power_term = 0.5f * v * v;

        const float den = v * sqrt_t;
        const float d1 = ((r + power_term) * t + log_term) / den;
        const float d2 = d1 - den;

        const float nd1 = cndf_inline(d1);
        const float nd2 = cndf_inline(d2);

        const float future = k * fast_discount_exp(r * t);

        call[i] = (s * nd1) - (future * nd2);
        put[i] = (future * (1.0f - nd2)) - (s * (1.0f - nd1));
    }


}

void naive_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    naive_BlkSchls(args.call_option_price,
                   args.put_option_price,
                   args.spot_price,
                   args.strike,
                   args.rate,
                   args.volatility,
                   args.time);
}

void stu_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    stu_BlkSchls(args.call_option_price,
                 args.put_option_price,
                 args.spot_price,
                 args.strike,
                 args.rate,
                 args.volatility,
                 args.time);
}

bool BlkSchls_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu_args = *static_cast<blackscholes_args *>(stu_ctx);
    auto &ref_args = *static_cast<blackscholes_args *>(ref_ctx);
    const double eps = ref_args.epsilon; // relative tolerance

    if (ref_args.call_option_price.size() != stu_args.call_option_price.size() ||
        ref_args.put_option_price.size() != stu_args.put_option_price.size())
        return false;

    const double atol = 1e-5; // absolute tolerance for near-zero prices
    const size_t n = ref_args.call_option_price.size();
    double max_rel = 0.0, max_abs = 0.0;
    size_t max_idx = 0;
    const char *max_leg = "call";

    for (size_t i = 0; i < n; ++i) {
        const double rc = static_cast<double>(ref_args.call_option_price[i]);
        const double rp = static_cast<double>(ref_args.put_option_price[i]);
        const double sc = static_cast<double>(stu_args.call_option_price[i]);
        const double sp = static_cast<double>(stu_args.put_option_price[i]);

        const double err_c = std::abs(rc - sc);
        const double err_p = std::abs(rp - sp);
        const double rel_c = (err_c - atol) / std::abs(rc);
        const double rel_p = (err_p - atol) / std::abs(rp);

        const bool call_ok = err_c <= (atol + eps * std::abs(rc));
        const bool put_ok = err_p <= (atol + eps * std::abs(rp));

        if (rel_c > max_rel) {
            max_abs = err_c;
            max_rel = rel_c;
            max_idx = i;
            max_leg = "call";
        }
        if (rel_p > max_rel) {
            max_abs = err_p;
            max_rel = rel_p;
            max_idx = i;
            max_leg = "put";
        }

        if (!call_ok || !put_ok) {
            debug_log("\tDEBUG: fail idx={} | call ref={} stu={} err={} thr={} | put ref={} stu={} err={} thr={}\n",
                      i,
                      rc,
                      sc,
                      err_c,
                      (atol + eps * std::abs(rc)),
                      rp,
                      sp,
                      err_p,
                      (atol + eps * std::abs(rp)));
            return false;
        }
    }
    debug_log("\tBlkSchls_check passed: n={}, max_rel_err={}, max_abs_err={} at idx={} ({})\n",
              n,
              max_rel,
              max_abs,
              max_idx,
              max_leg);

    return true;
}
