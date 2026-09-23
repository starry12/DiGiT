# Prepared PA data

The upstream dataset is [OGB ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M): 111,059,956 nodes, 128 float32 features, 172 classes, with official splits of 1,207,179 / 125,265 / 214,338 examples. Dataset attribution and redistribution terms remain those of OGB and the upstream providers.

The selected graph preserves original multiedges, adds reverse edges and exactly one self-loop per node: 3,342,431,700 edges. Both systems use the same graph, random training orders and sampled evaluation traces. DiGiT uses the prepared g2 layout; BFS is disabled. The GPU cache is 4 GiB and the static CPU cache contains 11,105,992 rows.

The repository excludes graph arrays, feature files, checkpoints and SSD contents. The prepared server supplies read-only CSC arrays, g2 metadata and reordered features, cache row selections, random orders, validation/test traces, and bound SSD write/readback receipts. Downloading OGB alone does not create those inputs. This submission uses the prepared-server path; a complete fresh download-to-layout-to-SSD pipeline is not yet validated.

On a separate built deployment, an administrator must arrange the prepared profile and register exactly these read-only bindings:

- `data/` and `ssd_state/`;
- validation and test trace directories named by the selected protocol;
- the g2 SSD verification receipt named in the preserved data profile.

```bash
python3 scripts/bind_data.py --prepared-root /absolute/prepared/profile --pre-mounted --locations /absolute/locations.json --dry-run
python3 scripts/bind_data.py --prepared-root /absolute/prepared/profile --pre-mounted --locations /absolute/locations.json
```

This registers existing read-only mounts, checks identities and binds them to this package hash. It neither creates the mounts nor writes SSD data. `locations.json` is a string mapping following [external_paths.json](../configs/external_paths.json). Use direct read-only mounts: symlinks can violate canonical bundle paths in existing receipts. Native inputs are rehashed by `check`, and device identity/readback receipts are checked before workers run.

Protocol JSON bytes are retained so existing preparation hashes remain valid. Some receipt/data identifiers therefore retain dated names. They identify prepared inputs; the corresponding old code and experiment output directories are not included. Only compact preparation source hashes and the deterministic correctness oracle are retained.
