import numpy as np
import networkx as nx
from itertools import combinations


def build_consistency_graph(src_points, tgt_points, correspondences, threshold=0.1):
    """
    构建一致性图：如果两对 correspondence 距离变化一致，则连边
    """
    G = nx.Graph()
    n = len(correspondences)
    G.add_nodes_from(range(n))

    for i, j in combinations(range(n), 2):
        si, ti = correspondences[i]
        sj, tj = correspondences[j]

        ds = np.linalg.norm(src_points[si] - src_points[sj])
        dt = np.linalg.norm(tgt_points[ti] - tgt_points[tj])

        if abs(ds - dt) < threshold * ds:
            G.add_edge(i, j)

    return G


def find_maximum_clique(G):
    """
    使用 Bron-Kerbosch 算法找最大团（networkx 提供了接口）
    """
    cliques = nx.find_cliques(G)  # 生成所有极大团
    max_clique = max(cliques, key=len)
    return sorted(max_clique)


def local_search_max_clique(G, max_iter=1000):
    """
    局部搜索求最大团（启发式）
    """

    # Step 1: 初始化一个极大团（贪心构造）
    def greedy_maximal_clique():
        nodes = sorted(G.nodes, key=lambda x: -G.degree[x])  # 度大优先
        clique = []
        for node in nodes:
            if all(G.has_edge(node, v) for v in clique):
                clique.append(node)
        return set(clique)

    best_clique = greedy_maximal_clique()
    current_clique = best_clique.copy()

    for _ in range(max_iter):
        # Step 2: 尝试添加一个外部顶点（与当前团全连接）
        candidates = [v for v in G.nodes if v not in current_clique and all(G.has_edge(v, u) for u in current_clique)]
        if candidates:
            # 贪心选度最高的
            v = max(candidates, key=G.degree)
            current_clique.add(v)
            if len(current_clique) > len(best_clique):
                best_clique = current_clique.copy()
            continue

        # Step 3: 如果无法添加，则尝试替换：删一个、加一个
        improved = False
        for u in current_clique:
            # 移除 u
            temp_clique = current_clique - {u}
            # 找能连接到 temp_clique 的外部顶点
            neighbors_of_temp = (
                set.intersection(*[set(G.neighbors(v)) for v in temp_clique]) if temp_clique else set(G.nodes)
            )
            candidates = neighbors_of_temp - temp_clique
            if candidates:
                v = max(candidates, key=G.degree)  # 选度高的
                new_clique = temp_clique | {v}
                if len(new_clique) > len(current_clique):
                    current_clique = new_clique
                    improved = True
                    break

        if not improved:
            break  # 局部最优

    return best_clique


# 使用示例
src_pts = np.random.rand(100, 3)
tgt_pts = np.random.rand(100, 3)
corres = np.array([[i, i] for i in range(50)])  # 假设前 50 是正确匹配

G = build_consistency_graph(src_pts, tgt_pts, corres, threshold=0.08)
max_clique_indices = find_maximum_clique(G)

inlier_corres = corres[max_clique_indices]
print(f"Found inlier set of size {len(inlier_corres)}")
