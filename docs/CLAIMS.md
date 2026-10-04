# Evaluation scope

DiGiT is a reconstruction of the paper implementation. The prepared server provides the following experiments.

| Experiment | Workload | Claim |
|---|---|---|
| PA × GraphSAGE / GCN / GAT | Seed 0, 20 full epochs per system, validation each epoch, final test | Training speedup and observed test accuracy |
| PA / GraphSAGE ablation | Four arms, one complete epoch each | Component performance at this configuration |
| PA / GraphSAGE layout grid | 15 layouts, one complete epoch each | Performance relative to g2/r20 with shared proxy features |
| IG / GraphSAGE | Five paired short windows | Maximum same-round speedup |
| UKS / GraphSAGE | Five paired short windows with synthetic features and labels | Maximum same-round speedup |

PA main settings: batch 1024, fanouts 10/5/5, three layers, hidden width 128, dropout 0.2, g2, GPU cache 4 GiB and CPU cache 11,105,992 rows. SAGE uses Adam lr 0.001 / weight decay 0; GCN/GAT use lr 0.01 / weight decay 0.001. GAT has four heads. Selected protocols are in `training/{sage,gcn,gat}/protocol.json`.

Single-seed test accuracy does not establish statistical equivalence. Short windows do not establish full-epoch or mean performance. Layout-grid proxy features do not support accuracy claims. For ablation, GR→NS changes feature layout as well as sampling, and NS→DiGiT changes CPU hot-set selection and GPU replacement together. The GIDS path includes a static CPU-cache adaptation.

UKL, CL, other systems and unaccepted experimental optimizations are outside this AE package. See [results](RESULTS.md) and each experiment's linked protocol for its timing denominator and evidence. The source-build and prepared-server workflows are distinct; CPU checks alone do not certify a new GPU/NVMe deployment.
