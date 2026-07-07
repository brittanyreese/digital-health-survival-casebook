# Contributing

A solo research project published as a methods casebook. These conventions keep the history clean enough for peer review and for others to reproduce the pipeline.

## Setup

```bash
uv sync   # runtime + dev deps
```

## Quality gates

Every change merged to `main` must pass:

```bash
uv run ruff check .                                # lint
uv run pyright                                     # type-check
uv run pytest                                      # tests
npx prettier@3.3.3 --check $(git ls-files '*.md')  # markdown prose-wrap
```

CI runs the same checks on every push. It also runs a reproducibility job on freshly generated seed-42 data. That job validates the tables and re-runs `tests/test_parameter_recovery.py` against the analysis outputs. Those tests score recovered statistics (such as the Weibull shape kappa) against the magnitudes injected into the seeded generator, not against internal consistency, so a change that breaks recovery fails CI even when the committed artifacts are untouched. CI does not assert byte-identical CSVs. Under seed 42 the tables are byte-identical on a fixed platform. Trailing digits can drift across platforms (macOS/ARM versus the Linux runner), so the job verifies recovery rather than byte equality.

## AI assistance

Built with AI coding tools, every change reviewed under the gates above. The maintainer makes the design decisions and validates results against the generator's injected parameters and the published sources.

## Commits

- Conventional Commits: `type(scope): subject` (subject <=72 chars). The body explains why.
- One logical change per commit. Keep formatting-only changes in their own `style:` commit. Never bundle a repo-wide reformat into a feature commit.
- New runtime dependency: justify it in the commit.

## Branching and releases

- Trunk-based: `main` stays green and releasable.
- Small changes go straight to `main`. Larger work uses a short-lived branch merged back promptly.
- Tag releases `vX.Y.Z`: include a `Release-version: X.Y.Z` trailer in the merge commit and `.github/workflows/release-tag.yml` cuts the tag automatically on merge to `main`. The version number is a human decision; only the tag-cutting is automated, so the tag cannot lag behind HEAD. For any reported result, cite the exact tagged commit that produced it.

## License

MIT (see [LICENSE](LICENSE)). Contributions are accepted under the same license.
