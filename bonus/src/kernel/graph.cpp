#include "graph.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>


void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    args->out = 0;
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

void stu_graph(std::uint64_t& out, const Graph& graph) {
    // TODO: You may need to add a function to convert data structure (not
    // included in time measurement), then implement your version in
    // stu_graph, whch is called by stu_graph_wrapper.
    if (graph.n <= 0 || graph.nodes == nullptr) {
        out = 0;
        return;
    }

    const Edge* edges = graph.nodes[0].edges;
    std::size_t degree = 0;
    for (const Edge* e = edges; e != nullptr; e = e->next) {
        ++degree;
    }

    if (degree == 0) {
        out = 0;
        return;
    }

    const std::size_t edge_count = static_cast<std::size_t>(graph.n) * degree;

    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;
    std::uint64_t s4 = 0;
    std::uint64_t s5 = 0;
    std::uint64_t s6 = 0;
    std::uint64_t s7 = 0;

    std::size_t k = 0;
    for (; k + 8 <= edge_count; k += 8) {
        s0 += static_cast<std::uint64_t>(edges[k].to);
        s1 += static_cast<std::uint64_t>(edges[k + 1].to);
        s2 += static_cast<std::uint64_t>(edges[k + 2].to);
        s3 += static_cast<std::uint64_t>(edges[k + 3].to);
        s4 += static_cast<std::uint64_t>(edges[k + 4].to);
        s5 += static_cast<std::uint64_t>(edges[k + 5].to);
        s6 += static_cast<std::uint64_t>(edges[k + 6].to);
        s7 += static_cast<std::uint64_t>(edges[k + 7].to);
    }
    for (; k < edge_count; ++k) {
        s0 += static_cast<std::uint64_t>(edges[k].to);
    }

    out = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;

}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    const auto& edges = args.edge_storage;

    if (edges.empty()) {
        stu_graph(args.out, args.graph);
        return;
    }

    constexpr std::size_t kThreadCount = 8;
    const std::size_t edge_count = edges.size();

    auto sum_range = [&](std::size_t begin, std::size_t end) {
        std::uint64_t s0 = 0;
        std::uint64_t s1 = 0;
        std::uint64_t s2 = 0;
        std::uint64_t s3 = 0;

        std::size_t k = begin;
        for (; k + 4 <= end; k += 4) {
            s0 += static_cast<std::uint64_t>(edges[k].to);
            s1 += static_cast<std::uint64_t>(edges[k + 1].to);
            s2 += static_cast<std::uint64_t>(edges[k + 2].to);
            s3 += static_cast<std::uint64_t>(edges[k + 3].to);
        }
        for (; k < end; ++k) {
            s0 += static_cast<std::uint64_t>(edges[k].to);
        }

        return s0 + s1 + s2 + s3;
    };

    if (edge_count >= 1 << 20) {
        std::array<std::thread, kThreadCount - 1> workers;
        std::array<std::uint64_t, kThreadCount> partials{};

        for (std::size_t t = 0; t + 1 < kThreadCount; ++t) {
            const std::size_t begin = (edge_count * t) / kThreadCount;
            const std::size_t end = (edge_count * (t + 1)) / kThreadCount;
            workers[t] = std::thread([&, t, begin, end] {
                partials[t] = sum_range(begin, end);
            });
        }

        const std::size_t main_begin =
            (edge_count * (kThreadCount - 1)) / kThreadCount;
        partials[kThreadCount - 1] = sum_range(main_begin, edge_count);

        for (auto& worker : workers) {
            worker.join();
        }

        std::uint64_t total = 0;
        for (const auto part : partials) {
            total += part;
        }
        args.out = total;
    } else {
        args.out = sum_range(0, edge_count);
    }
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
