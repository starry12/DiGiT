"""Stable CSC external sort with merge-by-merge durable scratch retirement."""
if __package__:
    from .large_preprocess import *
    from .retiring_job import Job
else:
    from large_preprocess import *
    from retiring_job import Job

if __package__:
    from .graph_source import POLICY
else:
    from graph_source import POLICY


def build_csc(edges, features, workspace, chunk_edges=262144, fan_in=16,
              memory_mib=256, stop_after=None):
    # Header preflight before expensive hashes or workspace creation.
    ef, ff = header(edges), header(features)
    validate_features(ff)
    n, e = ff["shape"][0], edge_count(ef)
    estimate = workspace_estimate(n, e + n, 1, 1, 1, 0, chunk_edges, fan_in)
    # Separate raw/loop runs can add a run at each merge level relative to
    # sorting one concatenated stream. Include conservative catalog slack.
    estimate["sort_checkpoint_files"] += 2 * ((e+n).bit_length()+1)
    check_heap(sort_heap_bound(chunk_edges, fan_in, estimate["sort_checkpoint_files"]), memory_mib)
    check_space(workspace, 48 * (e + n) + 8 * (3*n + e + 1) + 4096 * estimate["sort_checkpoint_files"])
    binding = dict(kind="stable_csc", graph_policy=POLICY,
                   graph_policy_sha256=digest(Path(__file__).with_name("graph_source.py")), edges=identity(edges), features=identity(features),
                   chunk_edges=chunk_edges, fan_in=fan_in, memory_mib=memory_mib,
                   code_sha256=digest(__file__), retirement_sha256=digest(Path(__file__).with_name("retiring_job.py")),
                   foundation_sha256=digest(Path(__file__).with_name("large_preprocess.py")), numpy=np.__version__)
    if any(binding["edges"][k] != v for k, v in ef.items()) or any(
            binding["features"][k] != v for k, v in ff.items()):
        raise ValueError("input header changed during preflight")
    with Job(workspace, binding, stop_after) as job:
        if job.state["phase"] == "complete":
            return job.state
        paths = []
        with Path(ef["path"]).open("rb") as source:
            for index, start in enumerate(range(0, max(e, 1), chunk_edges)):
                count = min(chunk_edges, e - start)
                def write_run(out, start=start, count=count):
                    components_contiguous = ((ef["shape"][0] == 2) != ef["fortran_order"])
                    if components_contiguous:
                        src = read_items(source, ef, start, count)
                        dst = read_items(source, ef, e + start, count)
                    else:
                        pairs = read_items(source, ef, 2 * start, 2 * count).reshape(-1, 2)
                        src, dst = pairs[:, 0], pairs[:, 1]
                    if count and (src.min() < 0 or src.max() >= n or dst.min() < 0 or dst.max() >= n):
                        raise ValueError("edge ID outside [0,N)")
                    keep = src != dst
                    run = np.empty(int(keep.sum()), dtype=RECORD)
                    run["src"], run["dst"] = src[keep], dst[keep]
                    run["eid"] = np.arange(start, start + count, dtype=np.int64)[keep]
                    order = np.argsort(dst[keep], kind="stable")
                    out.write(run[order].tobytes())
                paths.append(job.task("sort_00_{:08d}.bin".format(index), write_run))
        for index, start in enumerate(range(0, n, chunk_edges)):
            count = min(chunk_edges, n-start)
            def loops(out, start=start, count=count):
                ids = np.arange(start,start+count,dtype=np.int64)
                run = np.empty(count,dtype=RECORD)
                run["src"],run["dst"],run["eid"] = ids,ids,e+ids
                out.write(run.tobytes())
            paths.append(job.task("loops_{:08d}.bin".format(index),loops))
        level = 1
        while len(paths) > 1:
            next_paths = []
            for index, start in enumerate(range(0, len(paths), fan_in)):
                group = paths[start:start + fan_in]
                next_paths.append(job.task("sort_{:02d}_{:08d}.bin".format(level, index),
                                           lambda out, group=group: merge(group, out)))
                job.retire(group, [next_paths[-1]])
            paths, level = next_paths, level + 1
        ordered = paths[0]
        original_e = e
        e = job.state["files"][ordered.name]["bytes"] // 24

        def indices(out):
            npy_header(out, (e,), "<i8")
            buffer = bytearray()
            for _, _, src in records(ordered):
                buffer.extend(struct.pack("<q", src))
                if len(buffer) >= 65536:
                    out.write(buffer)
                    buffer.clear()
            out.write(buffer)

        def indptr(out):
            npy_header(out, (n + 1,), "<i8")
            owner, consumed, buffer = 0, 0, bytearray()
            for dst, _, _ in records(ordered):
                while owner <= dst:
                    buffer.extend(struct.pack("<q", consumed))
                    owner += 1
                    if len(buffer) >= 65536:
                        out.write(buffer)
                        buffer.clear()
                consumed += 1
            while owner <= n:
                buffer.extend(struct.pack("<q", consumed))
                owner += 1
                if len(buffer) >= 65536:
                    out.write(buffer)
                    buffer.clear()
            out.write(buffer)

        job.task("original_indices.npy", indices)
        job.task("original_indptr.npy", indptr)
        job.retire([ordered], [job.root / "original_indptr.npy", job.root / "original_indices.npy"])
        contract = dict(schema="digit-graph-source-v1", policy=POLICY,
                        original_edges=binding["edges"], num_nodes=n, num_edges=e,
                        removed_self_edges=original_e+n-e,
                        csc={name:job.state["files"][name] for name in ("original_indptr.npy","original_indices.npy")})
        job.task("graph_source.json",lambda out:out.write((json.dumps(contract,sort_keys=True,indent=2)+"\n").encode()))
        return job.complete(["original_indptr.npy", "original_indices.npy", "graph_source.json"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("edges", "features", "workspace"):
        p.add_argument("--" + name, required=True)
    for name, default in (("chunk-edges",16384),("fan-in",8),("memory-mib",64),("address-space-mib",256)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--stop-after-tasks",type=int)
    args=p.parse_args()
    address_space_limit(args.address_space_mib)
    build_csc(args.edges,args.features,args.workspace,args.chunk_edges,args.fan_in,args.memory_mib,args.stop_after_tasks)
