# Publication and submitted versions

The PA repository is already published at <https://github.com/starry12/DiGiT>. Continue with normal commits and pushes to the existing repository. The current concise source lives at `/home/embed/digit-ae-clean`; the older `/home/embed/digit-ae` wrapper repository is retained as history.

| Version | Commit | Meaning |
|---|---|---|
| Frozen `ae-pa-v1` | `e03c15277c57690fd7b7ee96acff85566d06b9c6` | Initial concise PA source snapshot |
| Documentation baseline before this update | `0b4cbad570215762fb563d9f377c61972cf0eb5b` | AE server workflow first, source setup later |

The frozen tag remains unchanged. If the submitted version should include newer documentation, create a new tag after committing and verifying that revision. Record its exact commit, archive checksum and result index. An archive of the initial tag is not an archive of later main.

## Updating the published source

1. Edit the concise checkout and update the affected file hashes in `ARTIFACT_MANIFEST.json`.
2. Run `bash run.sh verify` and review `git diff --check` and the full diff.
3. Commit the intended files and push normally to the existing `origin`. Do not rerun the initial history-replacement script or move `ae-pa-v1`.
4. Verify the remote revision and public download, then record the revision actually used by the AE form. Keep credentials and server access details private.

Do not commit datasets, credentials, checkpoints, local deployment bindings or raw generated run directories. Curated acceptance summaries and evidence hashes belong under `reference/` or `provenance/`. Source updates do not modify `/srv/digit-ae/releases/digit_ae_20260923_v3`; its native acceptance belongs to that sealed release. The concise source now has a fresh-install and compilation [receipt](../provenance/native_build_validation.json); PA/SAGE, GCN and GAT GPU/SSD preflight and paired smoke have also passed after the runtime dependency repair; the [rebuilt-source receipt](../reference/rebuilt_native_smoke.json) binds the accepted package, six worker results and 130 evidence hashes. The receipt records the original run identity; the documentation-only publication snapshot has its own manifest identity. These follow-up changes do not alter the submitted `ae-pa-v1`.

## Reviewer full results update — 2026-09-24

The current [result index](../reference/results.json) selects fresh reviewer full pairs for SAGE, GCN and GAT. [GCN/GAT acceptance](../reference/reviewer_full.json) records 202 evidence hashes; the [previous index](../reference/results_before_reviewer_full_20260924.json) preserves historical monitoring gaps and measurements. This documentation/evidence update changes no training implementation or sealed server package. The submitted `ae-pa-v1` remains unchanged.

## Submission closeout

The author confirmed on 2026-09-23 that the conference AE form has been submitted using `ae-pa-v1` (commit `e03c15277c57690fd7b7ee96acff85566d06b9c6`). This is an author-confirmed submission status; the conference success receipt was not independently inspected. Later main documentation updates do not change that submitted version. Preserve the tag and use a separate version for any subsequent delivery.

The project license decision remains pending in [LICENSE_STATUS](../LICENSE_STATUS.md) and must be made by the author. Final revision/archive/checksum indexing and long-term archiving remain separate closeout work.

## Supplementary ablation results update — 2026-09-24

This revision replaces the selected supplementary table with [cache-aligned results and adjacency-only GR](ABLATION.md): 194.45/194.76/161.73/104.50 s, 1.8608x. The previous published receipt and candidate identity are archived explicitly. Each current arm records its actual source candidate and report hash; GIDS/GR were reused, NS/DiGiT rerun. Main model rows, training implementation, reviewer service and frozen ae-pa-v1 remain unchanged. This update also supplies a separate prepared-server four-arm command extension; its original publication preceded installation and native acceptance. It does not modify the concise main training implementation.

The prepared-server extension is in `tools/ablation/`. Administrator installation is now verified, while the new four-arm native reproduction remains incomplete. At the author's request, retesting is deferred until a quieter server window; see [current ablation status](ABLATION.md#current-ae-reproduction-status--2026-09-24). The current published table remains the accepted author evidence.

The afternoon author measurements were first published in `b19f34d8ad784bf1b5afd3ddb1384ff0537c7412`. This update rechecks all 43 bound evidence files, adds a [full-precision CSV](../reference/pa_sage_ablation_perf.csv), and clarifies monitoring and pending AE reproduction. Training sources, selected measurements, main model results, and `ae-pa-v1` retain their existing identities.

## Four-arm monitor source synchronization

The updated four-arm controller uses the same external `nvidia-smi` monitor sources as the accepted afternoon author runs. Each query has a 5-second subprocess timeout, followed by a 0.5-second wait. The first successful sample is required before a worker starts. Later query errors are recorded while sampling continues; final acceptance requires zero query errors, a normally exited worker and monitor, valid ownership, and resource/report checks. The later NVML live heartbeat and phase-gap gates are removed. The host admission threshold stays at 192 GiB; cache sizes and training candidates are unchanged.

The source update is prepared and tested; administrator installation of this monitoring change is pending. The previously installed service still uses the recorded NVML policy until that installation succeeds. Retesting remains paused. The installer for this update performs a namespace selftest only and does not launch training. Monitor/source checks alone do not establish native acceptance. The source closure under `tools/ablation/` now includes the effective 192 GiB admission wrapper, fixed worker launcher and accepted-smoke validation helper. Private input bindings and reuse claims are not published. The selected author measurements and the immutable `ae-pa-v1` tag are unchanged.
