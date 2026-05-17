#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

void initialize_graph(graph_args* args,
                      std::size_t node_count,
                      int avg_degree,
                      std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(
        0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    args->graph_csr.n = 0;
    args->graph_csr.offsets.clear();
    args->graph_csr.edges.clear();

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

void convert_graph_to_csr(CSRGraph& dst, const Graph& src) {
    dst.n = src.n;
    dst.offsets.assign(static_cast<std::size_t>(src.n) + 1, 0);
    dst.edges.clear();

    if (src.n <= 0 || src.nodes == nullptr) {
        return;
    }

    std::size_t total_edges = 0;

    for (int u = 0; u < src.n; ++u) {
        dst.offsets[static_cast<std::size_t>(u)] =
            static_cast<int>(total_edges);

        for (const Edge* e = src.nodes[u].edges; e != nullptr; e = e->next) {
            ++total_edges;
        }
    }

    dst.offsets[static_cast<std::size_t>(src.n)] =
        static_cast<int>(total_edges);

    dst.edges.resize(total_edges);

    std::size_t pos = 0;
    for (int u = 0; u < src.n; ++u) {
        for (const Edge* e = src.nodes[u].edges; e != nullptr; e = e->next) {
            dst.edges[pos++] = e->to;
        }
    }
}

void stu_graph(std::uint64_t& out, const CSRGraph& graph) {
    const int n = graph.n;
    const int* offsets = graph.offsets.data();
    const int* edges = graph.edges.data();

    std::uint64_t checksum = 0;

    if (n <= 0 || offsets == nullptr || edges == nullptr) {
        out = 0;
        return;
    }

    for (int u = 0; u < n; ++u) {
        int begin = offsets[u];
        int end = offsets[u + 1];

        int i = begin;

        // Manual unrolling for the common avg_degree=8 case.
        for (; i + 7 < end; i += 8) {
            checksum += static_cast<std::uint64_t>(edges[i + 0]);
            checksum += static_cast<std::uint64_t>(edges[i + 1]);
            checksum += static_cast<std::uint64_t>(edges[i + 2]);
            checksum += static_cast<std::uint64_t>(edges[i + 3]);
            checksum += static_cast<std::uint64_t>(edges[i + 4]);
            checksum += static_cast<std::uint64_t>(edges[i + 5]);
            checksum += static_cast<std::uint64_t>(edges[i + 6]);
            checksum += static_cast<std::uint64_t>(edges[i + 7]);
        }

        for (; i < end; ++i) {
            checksum += static_cast<std::uint64_t>(edges[i]);
        }
    }

    out = checksum;
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    stu_graph(args.out, args.graph_csr);
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