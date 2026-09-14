## What changed and why

## How to test

```bash
make lint
make test-smoke   # or make test for the full fast suite
```

## Checklist

- [ ] Tests added/updated for new behaviour or bug fixes
- [ ] `make lint` passes
- [ ] Float64 invariant preserved (`jax_enable_x64` untouched, no float32 in hot path)
- [ ] Docs/examples updated if user-facing behaviour changed
- [ ] CHANGELOG.md entry added
