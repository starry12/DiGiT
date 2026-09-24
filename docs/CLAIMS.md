# Scope and claims

The DiGiT paper is the scope reference; original paper source was unavailable, so this implementation is a disclosed reconstruction. The submitted matrix is PA × GraphSAGE/GCN/GAT × GIDS/DiGiT.

| Paper topic | Experiment | Limit |
|---|---|---|
| Training time, §5.2.1 / Fig. 8 | Seed 0; 20 full epochs per system | Covers PA only, not the entire figure |
| Accuracy, §5.2.2 / Table 2 | One test of each newly trained epoch-20 model | Observed single-seed accuracy, not paper absolute-accuracy reproduction |
| Effective I/O | Useful and physical SSD bytes, active time and logical feature supply | Denominators are separate; combined system gain is not an ablation |
| Initial usability | CPU example, native preflight and paired smoke | CPU success alone is not native correctness/performance acceptance |

Fixed settings: batch 1024, fanouts 10/5/5, three layers, hidden width 128, dropout 0.2, g2, GPU cache 4 GiB and CPU cache 11,105,992 rows. SAGE uses Adam lr 0.001 / weight decay 0; GCN/GAT use lr 0.01 / weight decay 0.001. GAT has four heads. The authoritative selected protocols are in `training/{sage,gcn,gat}/protocol.json`.

IG/web graphs, ablations, sensitivity, scalability, other systems and multi-seed accuracy claims are excluded. The GIDS path includes the disclosed static CPU-cache adaptation. The source tree has been reorganized; current service evidence belongs to its separately sealed server release. [Source provenance](../provenance/source_map.json) records the relationship.

## Later supplementary evidence

The excluded-submission list above still describes the frozen AE scope. The later [cache-aligned author ablation](ABLATION.md) now includes adjacency-only GR and accepted NS/DiGiT full epochs. It supports this single-configuration, single-epoch performance table, not every paper ablation, isolated NS/FIFO attribution, multi-seed accuracy or native acceptance through the public CLI.
