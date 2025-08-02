import numpy as np
import numba as nb
from time import perf_counter
import torch


def ultra_fast_analysis(edge_path):
    print(f"\n[阶段1] 数据加载与验证")
    start_load = perf_counter()
    raw_data = np.load(edge_path)
    if raw_data.shape[0] == 2:
        edges = raw_data.T.astype(np.uint32)
    else:
        edges = raw_data[:, :2].astype(np.uint32)
    print(f"数据加载完成 | 形状：{edges.shape} | 内存占用：{edges.nbytes/1024**3:.1f}GB")
    print(f"首行样本：{edges[0]} | 末行样本：{edges[-1]}")
    print(f"加载耗时：{perf_counter() - start_load:.2f}s")

    print("\n[阶段2] 排序（正确顺序）")
    start_sort = perf_counter()
    sort_key = (edges[:, 1], edges[:, 0])
    sorted_order = np.lexsort(sort_key)
    sorted_edges = edges[sorted_order]
    print(f"排序完成 | 首行样本：{sorted_edges[0]}")
    print(f"排序耗时：{perf_counter() - start_sort:.2f}s")

    print("\n[阶段3] 邻接表构建")
    start_build = perf_counter()
    @nb.njit
    def compute_offsets(sorted_src, max_src_id):
        offsets = np.zeros(max_src_id + 2, dtype=np.uint32)
        current = sorted_src[0]
        count = 1
        for i in range(1, len(sorted_src)):
            if sorted_src[i] == current:
                count += 1
            else:
                offsets[current + 1] = count
                current = sorted_src[i]
                count = 1
        offsets[current + 1] = count
        for i in range(1, len(offsets)):
            offsets[i] += offsets[i - 1]
        return offsets
    src_nodes = sorted_edges[:, 0]
    max_src_id = int(src_nodes.max())
    offsets = compute_offsets(src_nodes, max_src_id)
    neighbors = sorted_edges[:, 1].astype(np.uint32)
    print(f"构建完成 | 总邻居数：{len(neighbors):,}")
    print(f"构建耗时：{perf_counter() - start_build:.2f}s")

    print("\n[阶段4] 统计信息生成")
    start_stat = perf_counter()
    actual_src_ids = np.unique(src_nodes)
    out_degrees = offsets[1:] - offsets[:-1]
    valid_degrees = out_degrees[actual_src_ids]
    stats = {
        "total_edges": len(edges),
        "unique_sources": int(len(actual_src_ids)),
        "max_out_degree": int(valid_degrees.max()),
        "avg_out_degree": float(valid_degrees.mean()),
        "median_out_degree": float(np.median(valid_degrees)),
        "out_degree_distribution": np.bincount(valid_degrees),
        "original_ids": actual_src_ids,
        "offsets": offsets,
        "neighbors": neighbors
    }
    print(f"统计耗时：{perf_counter() - start_stat:.2f}s")
    return stats

def add_self_edges_all(original_stats, edges, total_nodes=1_000_000):
    print("\n[自环处理-All] 初始化")
    start = perf_counter()
    node_count = total_nodes
    full_offsets = original_stats['offsets']
    if len(full_offsets) < node_count + 1:
        full_offsets = np.concatenate([full_offsets,
                                       np.full(node_count + 1 - len(full_offsets), full_offsets[-1], dtype=full_offsets.dtype)])
    full_neighbors = original_stats['neighbors']
    orig_counts = full_offsets[1:] - full_offsets[:-1]
    new_counts = orig_counts + 1
    new_offsets = np.empty(node_count + 1, dtype=np.uint32)
    new_offsets[0] = 0
    new_offsets[1:] = np.cumsum(new_counts)
    total_new_edges = int(new_offsets[-1])
    new_neighbors = np.empty(total_new_edges, dtype=np.uint32)
    print(f"节点总数：{node_count:,}，分配新邻居数组：{new_neighbors.size:,}")
    @nb.njit(parallel=True)
    def fill(full_offsets, full_neighbors, new_offsets, new_neighbors):
        n = len(new_offsets) - 1
        for i in nb.prange(n):
            s, e = full_offsets[i], full_offsets[i+1]
            for j in range(e - s):
                new_neighbors[new_offsets[i] + j] = full_neighbors[s + j]
            new_neighbors[new_offsets[i+1] - 1] = i
        return new_neighbors
    adjusted_neighbors = fill(full_offsets, full_neighbors, new_offsets, new_neighbors)
    print(f"填充耗时：{perf_counter() - start:.2f}s")
    new_degrees = new_counts
    final_stats = original_stats.copy()
    final_stats.update({
        'offsets': new_offsets,
        'neighbors': adjusted_neighbors,
        'total_edges': int(original_stats['total_edges'] + node_count),
        'max_out_degree': int(new_degrees.max()),
        'avg_out_degree': float(new_degrees.mean()),
        'median_out_degree': float(np.median(new_degrees)),
        'out_degree_distribution': np.bincount(new_degrees),
        'unique_sources': int(np.count_nonzero(new_degrees)),
        'original_ids': np.arange(node_count, dtype=np.uint32)
    })
    print(f"更新统计耗时：{perf_counter() - start:.2f}s")
    return final_stats

def split_neighbors_by_remove_indices_enhanced(stats, remove_indices):
    offsets = stats['offsets']
    neighbors = stats['neighbors']
    node_count = len(offsets) - 1
    mask = np.zeros(node_count, dtype=bool)
    mask[remove_indices] = True
    in_lists, out_lists = [], []
    for i in range(node_count):
        block = neighbors[offsets[i]:offsets[i+1]]
        in_lists.append(block[mask[block]])
        out_lists.append(block[~mask[block]])
    def build(lists):
        counts = np.array([len(x) for x in lists], dtype=np.uint32)
        offs = np.zeros(node_count+1, dtype=np.uint32)
        offs[1:] = np.cumsum(counts)
        flat = np.concatenate(lists).astype(np.uint32)
        return {
            'offsets': offs,
            'neighbors': flat,
            'original_ids': np.arange(node_count, dtype=np.uint32),
            'total_edges': int(flat.size),
            'max_out_degree': int(counts.max()) if counts.size else 0,
            'avg_out_degree': float(counts.mean()) if counts.size else 0.0,
            'median_out_degree': float(np.median(counts)) if counts.size else 0.0,
            'unique_sources': int(np.count_nonzero(counts))
        }
    return build(in_lists), build(out_lists)

@nb.njit
def _grouping_core_with_real_edges(order, offsets, neighbors, used, neighbor_groups, sources, group_count):
    for idx in range(len(order)):
        node = order[idx]
        if used[node]: continue
        s, e = offsets[node], offsets[node+1]
        cnt = 0
        for j in range(s, e):
            nbr = neighbors[j]
            if nbr < len(used) and not used[nbr]:
                neighbor_groups[group_count[0], cnt] = nbr
                cnt += 1
                if cnt == 4: break
        if cnt == 4:
            sources[group_count[0]] = node
            group_count[0] += 1
            for k in range(4):
                used[neighbor_groups[group_count[0]-1, k]] = 1

def full_optimized_grouping_with_real_edges(stats, prank_scores, redundancy_rate=0.2):
    offsets = stats['offsets'].astype(np.int32)
    neighbors = stats['neighbors'].astype(np.int32)
    total_nodes = len(offsets) - 1
    total_edges = neighbors.size
    degrees = offsets[1:] - offsets[:-1]
    print("\n==== [输入验证器] 初步分析图结构 ====")
    print(f"总节点数: {total_nodes:,}")
    print(f"总边数: {total_edges:,}")

    # 第1阶段：正常分组
    initial_nodes = total_nodes
    max_groups = total_edges // 4
    neighbor_groups = np.full((max_groups, 4), -1, dtype=np.int32)
    sources = np.full(max_groups, -1, dtype=np.int32)
    used = np.zeros(total_nodes, dtype=np.uint8)
    group_count = np.zeros(1, dtype=np.int32)
    order = np.lexsort((-prank_scores, degrees)).astype(np.int32)
    _grouping_core_with_real_edges(order, offsets, neighbors, used, neighbor_groups, sources, group_count)
    valid_ng = neighbor_groups[:group_count[0]]
    valid_src = sources[:group_count[0]]
    group_storage = valid_ng.size
    used_count = int(used.sum())
    missing_count = total_nodes - used_count
    first_storage = group_storage + missing_count
    print(f"[第一阶段] 有效组数量: {group_count[0]:,}")
    print(f"[第一阶段] 组内存储: {group_storage:,}, 未覆盖节点: {missing_count:,}")
    print(f"[第一阶段] 存储膨胀率: {first_storage/total_nodes:.2f}x")

    # 第2阶段：真实边冗余补组
    new_ng, new_src = [], []
    for node in np.where(used == 0)[0]:
        current_storage = group_storage + (total_nodes - used_count)
        if current_storage / total_nodes >= 1 + redundancy_rate:
            break
        nbrs = neighbors[offsets[node]:offsets[node+1]]
        if len(nbrs) >= 4:
            selected = nbrs[:4]
            new_ng.append(selected)
            new_src.append(node)
            group_storage += 4
            for nbr in selected:
                if not used[nbr]:
                    used[nbr] = 1
                    used_count += 1
    new_count = len(new_src)
    if new_count:
        new_ng = np.vstack(new_ng)
        new_src = np.array(new_src, dtype=np.int32)
    else:
        new_ng = np.empty((0,4), dtype=np.int32)
        new_src = np.empty((0,), dtype=np.int32)
    print(f"[第二阶段] 补充组数量: {new_count:,}")

    # 合并并统计
    all_ng = np.vstack([valid_ng, new_ng])
    all_src = np.concatenate([valid_src, new_src])
    group_nodes = all_ng.flatten()
    final_missing = total_nodes - used.sum()
    final_storage = group_nodes.size + final_missing
    print("\n==== 最终存储统计 ====")
    print(f"组内邻居节点数 (计重复): {group_nodes.size:,}")
    print(f"组外未覆盖节点数: {final_missing:,}")
    print(f"最终总存储节点数: {final_storage:,}")
    print(f"总存储膨胀率: {final_storage/total_nodes:.2f}x")

    return all_src, all_ng

def validate_groups_with_edges_allow_self_loop(all_sources, all_neighbor_groups, edges):
    edge_set = set((int(s), int(d)) for s, d in edges)
    total = len(all_sources)
    valid = 0
    for i in range(total):
        src = int(all_sources[i])
        nbrs = all_neighbor_groups[i]
        ok = True
        for d in nbrs:
            if src == d:
                continue
            if (src, int(d)) not in edge_set:
                ok = False
                break
        if ok:
            valid += 1
    print("\n==== 分组验证结果（允许自环） ====")
    print(f"总组数: {total:,}")
    print(f"验证通过的组数: {valid:,}")
    print(f"验证失败的组数: {total-valid:,}")
    return valid, total-valid

def build_edges_from_stats(stats):
    offsets, neighbors = stats['offsets'], stats['neighbors']
    total_edges = neighbors.size
    arr = np.empty((total_edges, 2), dtype=np.int32)
    idx = 0
    for src in range(len(offsets)-1):
        for j in range(offsets[src], offsets[src+1]):
            arr[idx] = [src, neighbors[j]]
            idx += 1
    return arr

def compress_edges_with_supernodes(edge_path, srcs, groups, total_nodes, out_path='edge_index_super.npy'):
    """
    用超级节点替换原始分组边（向量化实现）
    - 删除所有分组边
    - 添加每组 src->supernode 边
    """
    import numpy as _np
    raw = _np.load(edge_path)
    orig = raw.T if raw.shape[0]==2 else raw[:, :2]
    orig0 = orig[:,0].astype(_np.int64); orig1 = orig[:,1].astype(_np.int64)
    base = total_nodes + len(srcs)
    orig_keys = orig0 * base + orig1
    sd0 = _np.array(srcs, dtype=_np.int64)
    sd1 = _np.array(groups).flatten().astype(_np.int64)
    delete_keys = sd0.repeat(4) * base + sd1
    delete_unique = _np.unique(delete_keys)
    mask = _np.isin(orig_keys, delete_unique)
    kept = orig[~mask]
    super_ids = _np.arange(len(srcs), dtype=_np.int64) + total_nodes
    src_ids = _np.array(srcs, dtype=_np.int64)
    new_rows = _np.stack([src_ids, super_ids], axis=1)
    new_arr = _np.vstack([kept, new_rows]).astype(_np.int32)
    _np.save(out_path, new_arr)
    print(f"[压缩] 原始边数: {orig.shape[0]:,}, 新边数: {new_arr.shape[0]:,}")
    print(f"[压缩] 原始唯一源节点: {len(_np.unique(orig[:,0])):,}, 新唯一源节点: {len(_np.unique(new_arr[:,0])):,}")
    actual_deleted = mask.sum()
    expected_deleted = delete_unique.size
    print(f"[压缩] 删除分组边: 实际 {actual_deleted:,}, 目标 {expected_deleted:,}")
    if actual_deleted != expected_deleted:
        print(f"[警告] 删除差异: {expected_deleted - actual_deleted} 条")
    actual_added = new_rows.shape[0]
    print(f"[压缩] 添加超级节点边: 实际 {actual_added:,}, 目标 {len(srcs):,}")
    return new_arr

def reorder_node_features_from_array(feats, new_seq, out_feat_path, batch_size=100_000):
    """
    根据 new_node_sequence 数组重新排序节点特征，并保存为新的 .npy 文件。

    feats: 原始特征数组或 memmap，shape=(M, F)
    new_seq: 新节点序列索引数组，shape=(K,)，各元素需在 [0, M)
    out_feat_path: 输出特征文件路径（.npy）
    batch_size: 每批次处理的节点数量，默认 100k
    """
    M, F = feats.shape
    K = len(new_seq)
    print(f"[自定义阶段] 重排序节点特征: 原始节点={M:,}, 输出节点={K:,}, 特征维度={F}")

    # 为输出创建 memmap 文件，shape=(K, F)
    reordered = np.lib.format.open_memmap(
        out_feat_path,
        mode='w+',
        dtype=feats.dtype,
        shape=(K, F)
    )

    # 分批拷贝
    for start in range(0, K, batch_size):
        end = min(start + batch_size, K)
        reordered[start:end] = feats[new_seq[start:end]]
        #print(f"已处理节点特征: {start} 到 {end} / {K}")

    print(f"重排序完成，已保存至 {out_feat_path} | 形状: {reordered.shape}")
    return reordered


if __name__ == "__main__":
    print("==== 开始分析 ====")
    stats = ultra_fast_analysis('/home/embed/Downloads/igb_datasets/small/processed/paper__cites__paper/edge_index.npy')
    stats2 = add_self_edges_all(stats, None)
    remove_idx = torch.load('pr_small_30.pt').numpy()
    _, filtered = split_neighbors_by_remove_indices_enhanced(stats2, remove_idx)
    prank = torch.load('pr_small.pt').numpy()
    srcs, groups = full_optimized_grouping_with_real_edges(filtered, prank, redundancy_rate=0.2)
    raw = np.load('/home/embed/Downloads/igb_datasets/small/processed/paper__cites__paper/edge_index.npy')
    edges = raw.T if raw.shape[0]==2 else raw[:, :2]
    validate_groups_with_edges_allow_self_loop(srcs, groups, edges)
    # 生成新的节点序列和分组关系
    total_nodes = len(stats2['offsets']) - 1
    edges_sl = build_edges_from_stats(stats2)
    print(f"[自环] 构建后边数: {edges_sl.shape[0]:,}")
    np.save('edge_index_with_self.npy', edges_sl)
    compress_edges_with_supernodes('edge_index_with_self.npy', srcs, groups, total_nodes)

    grouped_nodes = groups.flatten()
    used_mask = np.zeros(total_nodes, dtype=bool)
    used_mask[grouped_nodes] = True
    remaining_nodes = np.where(~used_mask)[0]
    new_node_sequence = np.concatenate([grouped_nodes, remaining_nodes])
    grouping_relation = [(int(srcs[i]), [int(n) for n in groups[i]]) for i in range(len(srcs))]
    # 保存为文件
    #np.save('new_node_sequence.npy', new_node_sequence)
    #np.save('grouping_relation_srcs.npy', srcs)
    #np.save('grouping_relation_ngs.npy', groups)
    # 输出 new_node_sequence 的节点数目
    print(f"新节点序列长度: {len(new_node_sequence):,}")
    #print("已保存：new_node_sequence.npy, grouping_relation_srcs.npy, grouping_relation_ngs.npy")

    feats = np.load('/home/embed/Downloads/igb_datasets/small/processed/paper/node_feat.npy', mmap_mode='r')
    reorder_node_features_from_array(
        feats=feats,
        new_seq=new_node_sequence,
        out_feat_path='node_feat_reordered.npy'
    )
