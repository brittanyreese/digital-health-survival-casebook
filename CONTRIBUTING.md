# Contributing

A solo research project published as a methods casebook. These conventions keep the history clean enough for peer review and for others to reproduce the pipeline.

## Setup

```bash
uv sync   # runtime + dev deps
```

## Quality gates

Every change that lands on `main` must pass:

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format
uv run pyright               # type-check
uv run pytest                # tests
```

CI runs the same checks on every push, plus a full-pipeline reproducibility job that regenerates the synthetic tables under seed 42, runs the distributional validation suite, runs the full analysis pipeline, and then re-runs `tests/test_parameter_recovery.py` against the freshly generated outputs. Those tests score recovered statistics against the magnitudes injected into the seeded generator (for example the Weibull shape kappa), not against internal consistency, so a change that breaks recovery fails CI even when the committed artifacts are untouched. CI does not assert byte-identical CSVs: under seed 42 the result tables are byte-identical on a fixed platform, but trailing digits can drift across platforms (macOS/ARM vs the Linux runner), so the reproducibility job verifies recovery rather than byte equality.

## AI assistance

Built with AI coding tools, every change reviewed under the gates above. The maintainer makes the design decisions and validates results against the generator's injected parameters and the published sources.

## Commits

- Conventional Commits: `type(scope): subject` (subject <=72 chars; body explains the why).
- One logical change per commit. Keep formatting-only changes in their own `style:` commit; never bundle a repo-wide reformat into a feature commit.
- New runtime dependency: justify it in the commit.

## Branching and releases

- Trunk-based: `main` stays green and releasable.
- Small changes go straight to `main`. Larger work uses a short-lived branch merged back promptly.
- Tag releases `vX.Y.Z`. For any reported result, cite the exact tagged commit that produced it.

## License

MIT (see [LICENSE](LICENSE)). Contributions are accepted under the same license.
