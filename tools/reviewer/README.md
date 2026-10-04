# Reviewer entry

`cli.py` routes the supported commands directly to the prepared-server handlers. PA main inspection uses `pa_cli.py`; the other handlers belong to their installed service runtimes. The frontend changes command routing and result presentation, not training or resource protection.

Human-readable results show speedups and PA main test accuracy. Explicit `--json` preserves complete handler output. `--reference` selects an accepted reference for layout, IG and UKS; ordinary results always inspect the latest request.

```bash
python3 -B tools/reviewer/test_cli.py
```

These checks use mocked service calls and require no GPU, SSD or administrator privileges. Use the [reviewer workflow](../../docs/REVIEWER.md) on the supplied server.
