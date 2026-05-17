#include "trace_replay.h"

#include <algorithm>
#include <stdexcept>

namespace {

static inline uint64_t trace_replay_cost(const RequestRecord& record) {
    uint64_t cost = 0;
    cost += record.base_cost;
    cost += 2ull * record.retry_penalty;
    cost += record.miss_penalty;
    cost += record.bytes >> 4;
    return cost;
}

} // namespace

void initialize_trace_replay(trace_replay_args& args,
                             size_t record_count,
                             size_t trace_count,
                             uint32_t seed) {
    if (record_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: records must be non-empty.");
    }
    if (trace_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: trace must be non-empty.");
    }

    args.out = 0;
    args.records.resize(record_count);
    args.trace.resize(trace_count);

    uint32_t current = seed;

    for (size_t i = 0; i < args.records.size(); ++i) {
        current = current * 1664525u + 1013904223u;
        const uint32_t r0 = current;
        current = current * 1664525u + 1013904223u;
        const uint32_t r1 = current;

        args.records[i].base_cost = 20u + (r0 & 255u);
        args.records[i].retry_penalty = 1u + ((r0 >> 8) & 31u);
        args.records[i].miss_penalty = 1u + (r1 & 63u);
        args.records[i].bytes = 64u + ((r1 >> 8) & 511u);

        for (int k = 0; k < 24; ++k) {
            args.records[i].padding[k] =
                r0 ^ (r1 + static_cast<uint32_t>(k) * 17u);
        }
    }

    const uint32_t record_count_u32 = static_cast<uint32_t>(args.records.size());
    const uint32_t window_size = std::min<uint32_t>(1024u, record_count_u32);
    const uint32_t window_mask = window_size - 1;
    const uint32_t segment_len = 256u;
    const uint32_t window_count =
        std::max<uint32_t>(1u, record_count_u32 / window_size);

    uint32_t base = 0;
    uint32_t stride = 1;
    for (size_t i = 0; i < args.trace.size(); ++i) {
        if ((i % segment_len) == 0) {
            current = current * 1664525u + 1013904223u;
            base = (current % window_count) * window_size;

            current = current * 1664525u + 1013904223u;
            stride = ((current >> 3) & window_mask) | 1u;
        }

        const uint32_t local =
            static_cast<uint32_t>(i % segment_len) & window_mask;
        args.trace[i] = base + ((local * stride) & window_mask);
    }
}

void naive_trace_replay(uint64_t& out,
                        const std::vector<RequestRecord>& records,
                        const std::vector<uint32_t>& trace) {
    uint64_t total = 0;
    const uint64_t order_mix = 1315423911ull;

    for (size_t i = 0; i < trace.size(); ++i) {
        total = total * order_mix + trace_replay_cost(records[trace[i]]);
    }

    out = total;
}

void stu_trace_replay(uint64_t& out,
                      const std::vector<RequestRecord>& records,
                      const std::vector<uint32_t>& trace) {
    constexpr uint64_t order_mix = 1315423911ull;
    constexpr uint64_t order_mix2 = order_mix * order_mix;
    constexpr uint64_t order_mix3 = order_mix2 * order_mix;
    constexpr uint64_t order_mix4 = order_mix3 * order_mix;

    const size_t record_count = records.size();
    const size_t n = trace.size();

    static thread_local std::vector<uint32_t> costs;
    costs.resize(record_count);

    for (size_t i = 0; i < record_count; ++i) {
        const RequestRecord& record = records[i];
        costs[i] = static_cast<uint32_t>(
            static_cast<uint64_t>(record.base_cost) +
            2ull * record.retry_penalty +
            record.miss_penalty +
            (record.bytes >> 4)
        );
    }

    const uint32_t* cost_data = costs.data();
    const uint32_t* trace_data = trace.data();

    uint64_t total = 0;
    size_t i = 0;
    const size_t limit = n & ~size_t{3};

    for (; i < limit; i += 4) {
        const uint64_t c0 = cost_data[trace_data[i]];
        const uint64_t c1 = cost_data[trace_data[i + 1]];
        const uint64_t c2 = cost_data[trace_data[i + 2]];
        const uint64_t c3 = cost_data[trace_data[i + 3]];

        total = total * order_mix4 +
                c0 * order_mix3 +
                c1 * order_mix2 +
                c2 * order_mix +
                c3;
    }

    for (; i < n; ++i) {
        total = total * order_mix + cost_data[trace_data[i]];
    }

    out = total;
}

void naive_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    naive_trace_replay(args.out, args.records, args.trace);
}

void stu_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    stu_trace_replay(args.out, args.records, args.trace);
}

bool trace_replay_check(void* stu_ctx,
                        void* ref_ctx,
                        lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<trace_replay_args*>(stu_ctx);
    auto& ref_args = *static_cast<trace_replay_args*>(ref_ctx);
    return stu_args.out == ref_args.out;
}
