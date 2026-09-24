# Reading results

[Selected result JSON](../reference/results.json) contains one full paired reference per model with source hashes and monitoring qualifications. These are existing measured results, not a new run of this refactored checkout.

Training speedup is GIDS total training seconds divided by DiGiT total training seconds across 20 epochs. Report validation, test, preparation and total worker times separately. Final test accuracy uses each system's epoch-20 checkpoint; no test-set tuning is performed by the experiment.

Useful SSD throughput is useful SSD payload bytes divided by device active time. Physical SSD throughput uses transferred bytes with the same device denominator. Logical feature supply uses requested feature bytes divided by the feature-fetch interval. Overall training time also includes sampling, transfer and model computation; useful SSD throughput is not an end-to-end speedup.

The current index selects fresh reviewer full pairs for all three models, each with strict monitor acceptance and zero query timeouts. GCN/GAT completed on 2026-09-24 with 20 epochs, 20 full validations and one final test per arm; see the [full acceptance receipt](../reference/reviewer_full.json). Training speedups are 1.8584x and 1.7266x. DiGiT test accuracy is lower by 0.3788 and 0.6630 percentage points respectively; these differences are retained. The [previous index](../reference/results_before_reviewer_full_20260924.json) preserves the original GCN pair's four GIDS monitoring timeouts and the original GAT results. The [earlier smoke receipt](../reference/reviewer_smoke.json) and [rebuilt-source short acceptance](../reference/rebuilt_native_smoke.json) retain their own scope and package identities.

For a new local native run, use `summarize --input` with a successful matching-package invocation. The entry rejects old or failed runs as new results. `reference` explicitly displays the selected prior evidence. No statistical accuracy improvement or equivalence is claimed from one seed.
