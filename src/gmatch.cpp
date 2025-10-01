// gmatch.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace py = pybind11;
using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::Vector3f cross(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    return Eigen::Vector3f(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    );
}

/**
 * @brief calc distance error ratio, if error < thresh_geom_abs, return ratio; else return 1 (penalty)
 * @param matches: current matches
 * @param pairs: pairs to be evaluated
 * @param Me11: distance matrix between pts1
 * @param Me22: distance matrix between pts2
 */
std::vector<float> cost(
    const std::vector<std::pair<int, int>>& matches,
    const std::vector<std::pair<int, int>>& pairs,
    const Eigen::MatrixXf& Me11,
    const Eigen::MatrixXf& Me22,
    const float thresh_geom_abs
) {
    if (matches.empty()) return {};
    std::vector<float> costs(pairs.size(), 1.0f);

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        int i = pairs[idx].first;
        int j = pairs[idx].second;

        float max_err = 0.0f;
        float max_ratio = 0.0f;

        for (const auto& m : matches) {
            int m1 = m.first, m2 = m.second;
            float d1 = Me11(i, m1);
            float d2 = Me22(j, m2);
            float err = std::abs(d1 - d2);
            float ratio = (err + 1e-5f) / (1e-5f + d1);

            max_err = std::max(max_err, err);
            max_ratio = std::max(max_ratio, ratio);
        }

        if (max_err < thresh_geom_abs)
            costs[idx] = max_ratio;
    }
    return costs;
}


/**
 * @brief do flip-over check, return flags for each pair in `pairs`
 * @param matches: current matches
 * @param pairs: pairs to be evaluated
 * @param pts1: source points, (n1, 3), float
 * @param pts2: target points, (n2, 3), float
 * @param thresh_flip: threshold for flip-over detection
 * @return flags: true if flip-over detected, false otherwise
 */
std::vector<bool> flipover(
    const std::vector<std::pair<int, int>>& matches,
    const std::vector<std::pair<int, int>>& pairs,
    const Eigen::MatrixXf& pts1,
    const Eigen::MatrixXf& pts2,
    const float thresh_flip
) {
    if (matches.size() < 2) {
        return std::vector<bool>(pairs.size(), false);
    }

    std::vector<bool> flags(pairs.size(), false);
    Eigen::Vector3f v1_1 = pts1.row(matches[matches.size() - 2].first) - pts1.row(matches[matches.size() - 1].first);
    Eigen::Vector3f v2_1 = pts2.row(matches[matches.size() - 2].second) - pts2.row(matches[matches.size() - 1].second);

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        Eigen::Vector3f v1_2 = pts1.row(pairs[idx].first) - pts1.row(matches[matches.size() - 1].first);
        Eigen::Vector3f v2_2 = pts2.row(pairs[idx].second) - pts2.row(matches[matches.size() - 1].second);

        Eigen::Vector3f n1 = cross(v1_1, v1_2);
        Eigen::Vector3f n2 = cross(v2_1, v2_2);

        float n1_norm = n1.norm() + 1e-5;
        float n2_norm = n2.norm() + 1e-5;

        float n1_z_unit = n1[2] / n1_norm;
        float n2_z_unit = n2[2] / n2_norm;

        if (n1_z_unit * n2_z_unit < 0 && std::abs(n1_z_unit - n2_z_unit) > thresh_flip)
            flags[idx] = true; // flipover detected
    }
    return flags;
}

/**
 * @brief geometry-constrained keypoint matching; use branch-and-bound to generate initial hypotheses, then use greedy search to expand each hypothesis.
 * @param pts1: source points, (n1, 3), float
 * @param pts2: target points, (n2, 3), float
 * @param Mf12: feature distance matrix between pts1 and pts2, (n1, n2), float
 * @param thresh_feat: threshold for feature distance
 * @param L: max length of matches to be found
 * @param N_good: number of good initial pairs at branch-and-bound beginning
 * @param thresh_geom_ratio: threshold for geometric distance ratio
 * @param thresh_geom_abs: threshold for geometric distance absolute error
 * @param thresh_flip: threshold for flip-over detection
 * @return a pair of (matches, cost), where matches is a list of (i, j) index pairs, and cost is the final cost
 */
std::pair<std::vector<std::pair<int, int>>, float> gmatch_search_bnb(
    const MatrixXfRow& pts1,
    const MatrixXfRow& pts2,
    const MatrixXfRow& Mf12,
    float thresh_feat,
    int L = 24,
    int N_good = 24,
    float thresh_geom_ratio = 0.1f,
    float thresh_geom_abs = 0.005f,
    float thresh_flip = 0.8f
) {
    int n1 = pts1.rows(), n2 = pts2.rows();
    if (n1 == 0 || n2 == 0) return { {}, 1.0f };

    // Step 1: construct distance matrix Me11, Me22
    Eigen::MatrixXf Me11(n1, n1), Me22(n2, n2);
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n1; ++j)
            Me11(i, j) = (pts1.row(i) - pts1.row(j)).norm();
    for (int i = 0; i < n2; ++i)
        for (int j = 0; j < n2; ++j)
            Me22(i, j) = (pts2.row(i) - pts2.row(j)).norm();

    // Step 2: get candidate pairs
    std::vector<std::pair<int, int>> pairs_simi;
    std::vector<float> vec_f;
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
            if (Mf12(i, j) < thresh_feat) {
                pairs_simi.emplace_back(i, j);
                vec_f.push_back(Mf12(i, j));
            }

    if (pairs_simi.empty()) return { {}, 1.0f };

    // Step 3: get top-`N_good` similar feature pairs
    std::vector<int> indices(vec_f.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (indices.size() > N_good) {
        std::partial_sort(indices.begin(), indices.begin() + N_good, indices.end(), [&](int a, int b) { return vec_f[a] < vec_f[b]; });
        indices.resize(N_good);
    }

    std::vector<std::pair<int, int>> pairs_good;
    for (int idx : indices)
        pairs_good.push_back(pairs_simi[idx]);

    // Step 4: branch first
    std::vector<std::tuple<
        std::vector<std::pair<int, int>>,   // matches
        std::vector<std::pair<int, int>>,   // pairs
        std::vector<float>,                 // costs
        float                               // cost
        >> li;

    const int K = 8;
    for (auto& init_pair : pairs_good) {
        std::vector<std::pair<int, int>> matches = { init_pair };
        auto costs = cost(matches, pairs_simi, Me11, Me22, thresh_geom_abs);
        std::vector<std::pair<int, int>> filtered_pairs;
        std::vector<float> filtered_costs;

        for (size_t idx = 0; idx < pairs_simi.size(); ++idx) {
            if (costs[idx] < thresh_geom_ratio) {
                filtered_pairs.push_back(pairs_simi[idx]);
                filtered_costs.push_back(costs[idx]);
            }
        }

        if (filtered_pairs.empty()) continue;

        // get top-K
        std::vector<size_t> ind(filtered_costs.size());
        std::iota(ind.begin(), ind.end(), 0);
        int k = std::min(K, (int)ind.size());
        std::partial_sort(ind.begin(), ind.begin() + k, ind.end(),
            [&](size_t a, size_t b) { return filtered_costs[a] < filtered_costs[b]; });

        for (int i = 0; i < k; ++i) {
            size_t idx = ind[i];
            std::vector<std::pair<int, int>> new_matches = matches;
            new_matches.push_back(filtered_pairs[idx]);
            li.emplace_back(new_matches, filtered_pairs, filtered_costs, filtered_costs[idx]);
        }
    }

    // Step 5: branch second
    std::vector<std::tuple<
        std::vector<std::pair<int, int>>,
        std::vector<std::pair<int, int>>,
        std::vector<float>,
        float
        >> li2;

    for (auto& [matches, pairs, costs, c] : li) {
        auto new_costs = cost({ matches.back() }, pairs, Me11, Me22, thresh_geom_abs);
        std::vector<std::pair<int, int>> filtered_pairs;
        std::vector<float> filtered_costs;

        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            if (new_costs[idx] < thresh_geom_ratio) {
                filtered_pairs.push_back(pairs[idx]);
                filtered_costs.push_back(std::max(costs[idx], new_costs[idx]));
            }
        }

        if (filtered_pairs.empty()) continue;

        std::vector<size_t> ind(filtered_costs.size());
        std::iota(ind.begin(), ind.end(), 0);
        int k = std::min(K, (int)ind.size());
        std::partial_sort(ind.begin(), ind.begin() + k, ind.end(),
            [&](size_t a, size_t b) { return filtered_costs[a] < filtered_costs[b]; });

        for (int i = 0; i < k; ++i) {
            size_t idx = ind[i];
            std::vector<std::pair<int, int>> new_matches = matches;
            new_matches.push_back(filtered_pairs[idx]);
            li2.emplace_back(new_matches, filtered_pairs, filtered_costs, std::max(c, filtered_costs[idx]));
        }
    }

    const int N_hypo = N_good * K;

    // Step 6: get top-`N_hypo` (ranked by costs) as hypotheses
    if (li2.size() > N_hypo) {
        std::partial_sort(li2.begin(), li2.begin() + N_hypo, li2.end(),
            [](const auto& a, const auto& b) { return std::get<3>(a) < std::get<3>(b); });
        li2.resize(N_hypo);
    }

    // Step 7: dfs each hypothesis with greedy searching strategy
    std::vector<std::pair<int, int>> best_matches;
    float best_cost = 1.0f;

    for (auto& [matches, pairs, costs, c] : li2) {
        if (c >= thresh_geom_ratio) continue;

        while (matches.size() < L) {
            // 更新 cost
            auto new_costs = cost({ matches.back() }, pairs, Me11, Me22, thresh_geom_abs);
            std::vector<std::pair<int, int>> new_pairs;

            std::vector<float> new_costs_dp;
            for (size_t i = 0; i < pairs.size(); ++i) {
                float cost_dp = std::max(new_costs[i], costs[i]);
                if (cost_dp < thresh_geom_ratio) {
                    new_pairs.push_back(pairs[i]);
                    new_costs_dp.push_back(cost_dp);
                }
            }

            if (new_pairs.empty()) break;
            pairs = new_pairs;
            costs = new_costs_dp;

            // flip-over 检查
            auto flip_flags = flipover(matches, pairs, pts1, pts2, thresh_flip);

            std::vector<std::pair<int, int>> filtered_pairs;
            std::vector<float> filtered_costs;
            for (size_t i = 0; i < pairs.size(); ++i) {
                if (!flip_flags[i]) {
                    filtered_pairs.push_back(pairs[i]);
                    filtered_costs.push_back(costs[i]);
                }
            }

            if (filtered_pairs.empty()) break;

            size_t best_idx = 0;
            for (size_t i = 1; i < filtered_pairs.size(); ++i) {
                if (filtered_costs[i] < filtered_costs[best_idx]) {
                    best_idx = i;
                }
            }

            matches.push_back(filtered_pairs[best_idx]);
            c = std::max(c, filtered_costs[best_idx]);
        }

        if (matches.size() > best_matches.size() ||
            (matches.size() == best_matches.size() && c < best_cost)) {
            best_matches = matches;
            best_cost = c;
        }

        if (best_matches.size() >= L) break;
    }

    return { best_matches, best_cost };
}

PYBIND11_MODULE(gmatch_cpp, m) {
    m.doc() = "GMatch: Fast C++ implementation of feature matching with geometric constraints";

    m.def("gmatch_search_bnb", &gmatch_search_bnb, py::return_value_policy::copy,
        py::arg("pts1"), py::arg("pts2"), py::arg("Mf12"), py::arg("thresh_feat"), py::arg("L"), py::arg("N_good"), py::arg("thresh_geom_ratio"), py::arg("thresh_geom_abs"), py::arg("thresh_flip"),
        "Branch-and-bound search for geometric matching");
}