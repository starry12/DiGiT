# UKS/SAGE performance comparison

Use the [prepared-server commands](../../docs/UKS_PERFORMANCE.md) to reproduce the accepted experiment.

The service entry is `service/runner.py`. Controller, worker and review modules enforce the selected protocol and require successful worker exits and accepted result records. Supplementary runtime dependencies retain their internal import paths and identities; they are not alternative reviewer commands. The server supplies compiled components, input receipts and read-only datasets.

[Source map](../../docs/CODE.md) · [Reviewer workflow](../../docs/REVIEWER.md).
