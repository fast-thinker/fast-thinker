# Contributing

Thanks for helping improve Thinker. Keep changes focused and explain any
protocol or scoring impact.

## Local setup

```bash
uv python install 3.11
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff check .
```

Validator-only dependencies are GPU- and platform-sensitive. Install
`.[validator]` when exercising the full validator runtime; use `.[miner]` for
miner submission flows.

## Pull requests

- Describe user-visible, protocol, scoring, and security effects.
- Avoid unrelated formatting or dependency churn.
- Do not include generated adapters, model weights, datasets, wallet files,
  tokens, logs containing private evaluation data, or `.thinker/` state.
- Report suspected vulnerabilities privately as described in
  [SECURITY.md](SECURITY.md).
