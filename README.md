# Thinker Validator Miner

Thinker is a reasoning-model subnet where miners submit LoRA adapters for a
fixed base model, and validators score those adapters on math, long-context QA,
and multiple-choice reasoning tasks. The subnet rewards adapters that improve
reasoning quality while keeping submissions compact and validator evaluation
repeatable.

New or updated miner models mature on chain for six epochs (approximately
eight hours) before validators include them in evaluation. Publishing another
submission resets the waiting period.

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/) or another PEP 517-compatible installer
- A Bittensor wallet and hotkey for live subnet use
- A CUDA-capable Linux host for validator inference

## Install

```bash
cd thinker-validator-miner
uv python install
uv venv
source .venv/bin/activate
```

Miner:

```bash
uv pip install -e ".[miner]"
btcli --version
```

Validator:

```bash
uv pip install -e ".[validator]"
btcli --version
```

## Miner Submit

Set a Hugging Face write token:

```bash
export HF_TOKEN=hf_write_token
```

Submit a LoRA adapter directory containing `adapter_config.json` and
`adapter_model.safetensors`:

```bash
thinker-miner submit \
  --adapter-dir ./adapters/run-001 \
  --hf-repo hf-user/submissions \
  --wallet miner-wallet \
  --hotkey miner-hotkey \
  --network finney
```

CLI-required inputs: `--adapter-dir`, `--hf-repo`, and `HF_TOKEN` or
`--hf-token`. Set `--wallet`, `--hotkey`, and `--network` explicitly for a real
subnet run. `--netuid` defaults to `16`.

Adapter bundles default to a `500 MiB` size limit, shared by miner submission
and validator validation. Set `THINKER_MAX_ADAPTER_BYTES` on both sides only if
the subnet policy intentionally changes.

During submit, the CLI shows validators with valid encryption keys and requires
an explicit recipient choice. Type `all` for every listed validator or enter a
comma-separated UID list such as `1,2,3`. There is no default choice: blank or
invalid interactive input is rejected and the CLI asks again. For
non-interactive submissions, provide the choice with `--validator-uids`.

## Validator Run

Set the required W&B key:

```bash
export WANDB_API_KEY=wandb_api_key
```

Run validator scoring:

```bash
thinker-validator run \
  --wallet validator-wallet \
  --hotkey validator-hotkey \
  --evaluation-delay-epochs 6 \
  --burn-rate 0.9
```

Required runtime input: `WANDB_API_KEY`. Set `--wallet` and `--hotkey` only when
you are not using the local defaults. Subnet constants such as network, netuid,
owner hotkey, shared W&B project, model revision, and retrieval defaults are
configured in code.

`--evaluation-delay-epochs` defaults to `6`. Set it to `0` to disable the
maturity delay for local testing. While submissions mature, the validator logs
their remaining blocks and eligible epoch instead of reporting that no miner
submissions exist.

`--burn-rate` defaults to `0`. For example, `--burn-rate 0.9` assigns weight
`0.9` to burn UID 0 and distributes the remaining `0.1` among scored miners in
proportion to their scores. The accepted range is `0` through `1`, inclusive.

Evaluation scores are averaged within each task type before the task types are
combined. The default task weights are math `0.50`, long-context QA `0.30`, and
multiple-choice `0.20`; set `THINKER_SCORE_WEIGHT_MATH`,
`THINKER_SCORE_WEIGHT_LONG_CONTEXT_QA`, or
`THINKER_SCORE_WEIGHT_MULTIPLE_CHOICE` to override them.

## Validator Chat

Test a miner adapter from the validator side:

```bash
thinker-validator chat --miner 12 \
  --wallet validator-wallet \
  --hotkey validator-hotkey
```

By default each prompt is stateless. Add `--history` to keep prior
user/assistant turns in context during the chat session.

## Help

```bash
thinker-miner submit --help
thinker-validator run --help
thinker-validator chat --help
```

## Development

Install the package with its development tools, then run the local checks:

```bash
uv pip install -e ".[dev]"
ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution expectations and
[SECURITY.md](SECURITY.md) for private vulnerability reporting. Never commit
wallet material, Hugging Face or W&B tokens, generated adapters, or `.thinker/`
runtime state.

## Design

[proposal.md](proposal.md) describes the subnet protocol, scoring model, and
submission-encryption design. It is design documentation rather than a
compatibility guarantee; the implementation and on-chain configuration remain
authoritative.
