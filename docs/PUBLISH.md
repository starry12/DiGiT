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

## Submission closeout

The author confirmed on 2026-09-23 that the conference AE form has been submitted using `ae-pa-v1` (commit `e03c15277c57690fd7b7ee96acff85566d06b9c6`). This is an author-confirmed submission status; the conference success receipt was not independently inspected. Later main documentation updates do not change that submitted version. Preserve the tag and use a separate version for any subsequent delivery.

The project license decision remains pending in [LICENSE_STATUS](../LICENSE_STATUS.md) and must be made by the author. Final revision/archive/checksum indexing and long-term archiving remain separate closeout work.
