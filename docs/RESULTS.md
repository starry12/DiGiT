# Reading results

[Selected result JSON](../reference/results.json) contains one full paired reference per model with source hashes and monitoring qualifications. These are existing measured results, not a new run of this refactored checkout.

Training speedup is GIDS total training seconds divided by DiGiT total training seconds across 20 epochs. Report validation, test, preparation and total worker times separately. Final test accuracy uses each system's epoch-20 checkpoint; no test-set tuning is performed by the experiment.

Useful SSD throughput is useful SSD payload bytes divided by device active time. Physical SSD throughput uses transferred bytes with the same device denominator. Logical feature supply uses requested feature bytes divided by the feature-fetch interval. Overall training time also includes sampling, transfer and model computation; useful SSD throughput is not an end-to-end speedup.

The fresh reviewer SAGE full pair has strict monitor acceptance. GCN's selected original full pair had four GIDS monitoring query timeouts and remains qualified accordingly. GAT's original full pair passed strict monitoring. Fresh reviewer GCN and GAT paired smoke now passed with no monitor query errors and normal resource release; [the smoke receipt](../reference/reviewer_smoke.json) records both arms. These checks remain separate from the full 20-epoch measurements above.

For a new local native run, use `summarize --input` with a successful matching-package invocation. The entry rejects old or failed runs as new results. `reference` explicitly displays the selected prior evidence. No statistical accuracy improvement or equivalence is claimed from one seed.
