# UKS/SAGE performance comparison

| Reference | Speedup vs GIDS |
|---|---:|
| AE prepared-server replay | **2.00×** |

The prepared-server AE replay passed: the short check and all ten formal workers
exited normally, and the service completed successfully. This comparison measures performance with
real sampling, SSD I/O and model updates using synthetic features and labels;
it does not report accuracy or full-epoch training.

## Reproduction

```bash
source /srv/digit-ae/activate.sh
digit-ae performance UKS sage
digit-ae status UKS sage --action performance
digit-ae results UKS sage --action performance
digit-ae logs UKS sage --action performance
```

The service selects an idle GPU from cards 0–3 and uses the same card throughout
the request. It runs a short check followed by five paired rounds, each with
20 warmup and 300 timed mini-batches per system. If no GPU is available, it exits
without starting training. Requests continue after SSH disconnects.

To inspect the accepted AE reference without starting a run:
`digit-ae results UKS sage --action performance --reference`.
To cancel: `digit-ae stop UKS sage --action performance`.

Default results show the final speedup. `PASS` requires successful service
completion and accepted worker reports; `AE_REFERENCE` is separate from a
new execution. Measurement definitions are retained in the
[result record](../reference/uks_sage_performance.json); implementation and
configuration are in [the source extension](../tools/uks_performance/README.md).
See also the [installation receipt](../reference/uks_sage_server_installation.json).

[AE acceptance receipt](../reference/uks_sage_reviewer_acceptance.json).
