from __future__ import annotations

import argparse
import logging
import random
import sys
import threading
import time
import os
from pathlib import Path

from thinker.config import NETUID, ThinkerConfig, load_config
from thinker.retrieval.bm25 import (
    BM25RetrievalService,
    BM25S_PREBUILT_INDEX_REPO,
    CorpusFormatError,
    download_prebuilt_bm25_index,
    download_s3_wiki18_corpus,
)
from thinker.retrieval.server import RetrievalServiceHandle, start_retrieval_service

logger = logging.getLogger(__name__)

_DEFAULT_KEY_PATH = str(Path.home() / ".thinker" / "validator" / "enc_key.hex")
_DEFAULT_DECONTAM_PATH = str(Path.home() / ".thinker" / "validator" / "seen.txt")
_MOCK_MINER_ID = "mock-miner-1"
_EMPTY_ROUND_RETRY_SECONDS = 60
# Bittensor enforces a minimum block interval between weight-sets per hotkey
# (weights_rate_limit), independent of how often the validator re-scores
# miners -- so weight-setting runs on its own background-thread cadence
# (chain.PeriodicWeightSetter) decoupled from the (much slower) evaluation
# loop, rather than writing to chain inline after every epoch. A dedicated
# timer thread retries on a fixed interval and logs-and-continues on failure
# (for example, a rate-limit rejection) instead of checking chain state up
# front or crashing the process.
_WEIGHT_RETRY_SECONDS = 600
_OWNER_KEY_REFRESH_SECONDS = 15 * 60
_VALIDATOR_INFERENCE_METHODS = (
    "generate_original",
    "generate_original_limited",
    "generate_original_greedy_limited",
    "generate_original_samples",
    "generate_limited",
    "generate_for_miners_batch",
    "count_tokens",
    "suppress_progress",
)


def _status(message: str) -> None:
    print(f"[thinker-validator] {message}", flush=True)


def _limit(value: int) -> int | None:
    return value if value > 0 else None


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _resolve_paths(config: ThinkerConfig, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cache_dir = Path(args.retrieval_cache_dir or config.retrieval_cache_dir)
    corpus_raw = args.corpus_path or config.retrieval_corpus_path
    corpus_path = Path(corpus_raw) if corpus_raw else cache_dir / "wiki-18.jsonl.gz"
    index_raw = args.index_dir or config.retrieval_index_dir
    index_dir = Path(index_raw) if index_raw else cache_dir / "bm25s-wiki18"
    return cache_dir, corpus_path, index_dir


def prepare_retriever(config: ThinkerConfig, args: argparse.Namespace) -> BM25RetrievalService:
    cache_dir, corpus_path, index_dir = _resolve_paths(config, args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    explicit_corpus = bool(args.corpus_path or config.retrieval_corpus_path)
    auto_download = config.retrieval_auto_download and not args.no_auto_download
    _status(f"retrieval cache: {cache_dir}")
    _status(f"retrieval index: {index_dir}")

    if (
        not args.rebuild_index
        and not explicit_corpus
        and auto_download
        and not BM25RetrievalService.index_exists(index_dir)
    ):
        _status(
            f"retrieval index missing; downloading prebuilt BM25 index from "
            f"{BM25S_PREBUILT_INDEX_REPO}. This is large. Use --no-retrieval "
            "to skip retrieval during validator startup."
        )
        try:
            download_prebuilt_bm25_index(index_dir)
            _status("prebuilt BM25 index download complete")
        except Exception as exc:
            logger.warning(
                "could not download prebuilt bm25 index from %s (%s); "
                "falling back to building it from the raw corpus",
                BM25S_PREBUILT_INDEX_REPO,
                exc,
            )
            _status(
                "prebuilt BM25 index download failed; falling back to raw corpus indexing"
            )

    if not corpus_path.exists() and not BM25RetrievalService.index_exists(index_dir):
        if not auto_download:
            raise FileNotFoundError(
                f"retrieval corpus not found at {corpus_path}; set THINKER_RETRIEVAL_CORPUS_PATH "
                "or run without --no-auto-download"
            )
        _status(
            f"retrieval corpus missing; downloading {corpus_path.name}. "
            "This can take a while."
        )
        corpus_path = download_s3_wiki18_corpus(cache_dir)
        _status(f"retrieval corpus ready: {corpus_path}")

    def _build(path: Path) -> BM25RetrievalService:
        will_load = (
            not args.rebuild_index and BM25RetrievalService.index_exists(index_dir)
        )
        if will_load:
            _status(
                f"loading BM25 retrieval index from {index_dir} "
                f"({'mmap' if config.retrieval_mmap_index and not args.no_mmap else 'memory'})"
            )
        else:
            _status(
                f"building BM25 retrieval index from {path}; this may be slow"
            )
        retriever = BM25RetrievalService.build_or_load(
            corpus_path=path,
            index_dir=index_dir,
            limit=_limit(args.corpus_limit if args.corpus_limit is not None else config.retrieval_corpus_limit),
            mmap=config.retrieval_mmap_index and not args.no_mmap,
            force_rebuild=args.rebuild_index,
        )
        _status(f"retrieval index ready with {len(retriever.documents)} document(s)")
        return retriever

    try:
        return _build(corpus_path)
    except CorpusFormatError:
        if explicit_corpus or args.no_auto_download or not config.retrieval_auto_download:
            raise
        _status("default retrieval corpus is corrupt; re-downloading it")
        corpus_path = download_s3_wiki18_corpus(cache_dir, force_download=True)
        return _build(corpus_path)


def start_validator_runtime(config: ThinkerConfig, args: argparse.Namespace) -> RetrievalServiceHandle:
    _status("bootstrapping local retrieval service")
    retriever = prepare_retriever(config, args)
    _status("starting retrieval HTTP server")
    handle = start_retrieval_service(
        retriever,
        host=args.retrieval_host or config.retrieval_host,
        port=args.retrieval_port if args.retrieval_port is not None else config.retrieval_port,
        default_topk=(
            args.retrieval_default_topk
            if args.retrieval_default_topk is not None
            else config.retrieval_default_topk
        ),
    )
    _status(f"retrieval HTTP server listening at {handle.url}/retrieve")
    return handle


def _add_retrieve_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "retrieve",
        help="Bootstrap the local BM25 retrieval microservice (long-context QA support).",
    )
    _add_retrieval_arguments(parser)
    parser.add_argument(
        "--exit-after-bootstrap",
        action="store_true",
        help="Build/load index and start retrieval once, then stop. Useful for smoke tests.",
    )


def _add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-path", default=None, help="Path to S3 wiki-18.jsonl or wiki-18.jsonl.gz")
    parser.add_argument("--index-dir", default=None, help="Directory for the cached bm25s index")
    parser.add_argument("--retrieval-cache-dir", default=None, help="Cache root for corpus and index")
    parser.add_argument("--retrieval-host", default=None, help="Retrieval bind host")
    parser.add_argument("--retrieval-port", type=int, default=None, help="Retrieval bind port")
    parser.add_argument("--retrieval-default-topk", type=int, default=None)
    parser.add_argument("--corpus-limit", type=int, default=None, help="Testing only: index first N docs")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding bm25s index")
    parser.add_argument("--no-auto-download", action="store_true", help="Do not download S3 corpus if missing")
    parser.add_argument("--no-mmap", action="store_true", help="Load cached bm25s index fully into memory")


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-model-path", default=None,
        help="Local path or Hugging Face repo id for the frozen base model "
        "(default: config.base_model_repo)",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face read token (default: $HF_TOKEN)")
    parser.add_argument("--max-loras", type=int, default=4, help="Max concurrent vLLM LoRA slots")
    parser.add_argument(
        "--temperature",
        type=_positive_float,
        default=1.0,
        help="Sampling temperature for validator scoring generation (default: 1.0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Generation-only cap; final generation cap is also limited by --max-total-tokens",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=65536,
        help="Maximum chat prompt + generated tokens per request (default: 65536)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=65536,
        help="vLLM max_model_len for long-context runs (default: 65536)",
    )
    parser.add_argument(
        "--no-generation-progress",
        action="store_true",
        help="Disable generation progress bars.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Max concurrent sequences vLLM schedules per step -- raise this when batching "
        "many miners' requests into one generate() call so the engine doesn't artificially "
        "serialize a wide batch.",
    )
    parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching (on by default). Long-context QA sends the same "
        "system prompt/reference documents to every miner on a given question, so prefix "
        "caching lets vLLM reuse that shared KV cache instead of recomputing it per miner.",
    )


def _add_chat_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "chat",
        help="Open an interactive chat against one miner UID's current LoRA adapter.",
    )
    parser.add_argument("--miner", type=int, required=True, help="Miner UID to chat with")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Keep prior user/assistant turns in the chat context. Default is stateless.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to prepend to each chat request.",
    )
    parser.add_argument("--wallet", default="default", help="Bittensor wallet name")
    parser.add_argument("--hotkey", default="default", help="Bittensor hotkey name")
    parser.add_argument("--netuid", type=int, default=NETUID, help="Subnet netuid")
    parser.add_argument("--network", default="finney", help="Bittensor network: finney, test, or local")
    parser.add_argument(
        "--key-path", default=_DEFAULT_KEY_PATH,
        help="Path to this validator's persistent encryption keypair file",
    )
    _add_inference_arguments(parser)
    return parser


def _add_run_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "run", help="Run the validator epoch loop: score miner submissions and set weights."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run live evaluation with all real submissions plus one mock miner "
            "cloned from a random valid real submission."
        ),
    )
    parser.add_argument("--wallet", default="default", help="Bittensor wallet name")
    parser.add_argument("--hotkey", default="default", help="Bittensor hotkey name")
    parser.add_argument("--netuid", type=int, default=NETUID, help="Subnet netuid")
    parser.add_argument("--network", default="finney", help="Bittensor network: finney, test, or local")
    parser.add_argument(
        "--no-set-weights",
        action="store_true",
        help="Run the epoch loop and compute scores without writing weights to chain. "
        "Useful for local testing with an unregistered hotkey.",
    )
    parser.add_argument(
        "--burn-rate",
        type=_unit_interval_float,
        default=0.0,
        help=(
            "Fraction of on-chain weight assigned to burn UID 0; the remainder "
            "is distributed among scored miners (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--key-path", default=_DEFAULT_KEY_PATH,
        help="Path to this validator's persistent encryption keypair file",
    )
    parser.add_argument(
        "--decontam-path", default=_DEFAULT_DECONTAM_PATH,
        help="Path to the decontamination store's history file",
    )
    _add_inference_arguments(parser)
    parser.add_argument(
        "--common-seed-repo",
        default=None,
        help="Public Hugging Face dataset repo containing owner-encrypted common seeds "
        "(default: THINKER_COMMON_SEED_REPO).",
    )
    parser.add_argument(
        "--long-context-qa-per-epoch",
        type=int,
        default=None,
        help="Number of long-context QA samples per epoch (default: THINKER_N_LONG_CONTEXT_QA_PER_EPOCH)",
    )
    parser.add_argument(
        "--no-long-context-qa",
        action="store_true",
        help="Disable long-context QA samples for this run.",
    )
    parser.add_argument(
        "--test-mode",
        choices=("math", "long_qa", "science"),
        default=None,
        help=(
            "Run one task-only full-evaluation pass: math, long_qa, or science "
            "(multiple-choice). Test mode skips W&B logging and chain weights."
        ),
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=12.0, help="Seconds between checks for a new epoch"
    )
    parser.add_argument(
        "--evaluation-delay-epochs",
        type=_non_negative_int,
        default=None,
        help=(
            "Number of full epochs a new or updated miner submission must mature "
            "before evaluation (default: 6; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--once", action="store_true", help="Run exactly one epoch then exit. Useful for smoke tests."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Stop after this many epochs (default: run forever)"
    )
    parser.add_argument(
        "--wandb-project", default=None, help="Wandb project, shared across all validators (default: config.wandb_project)"
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="Wandb entity/team (default: config.wandb_entity, or the API key's default)"
    )
    _add_retrieval_arguments(parser)
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Do not bootstrap the local retrieval service before scoring.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thinker-validator",
        description="Run the Thinker validator: score miner submissions and set weights, "
        "or bootstrap the local retrieval microservice.",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_retrieve_subparser(subparsers)
    _add_run_subparser(subparsers)
    _add_chat_subparser(subparsers)
    return parser


def _serve_until_signalled(on_stop) -> None:
    try:
        while True:
            time.sleep(1)
    finally:
        on_stop()


def _run_retrieve(args: argparse.Namespace) -> int:
    config = load_config()
    handle = start_validator_runtime(config, args)
    print(f"Thinker retrieval service ready at {handle.url}/retrieve", flush=True)
    if args.exit_after_bootstrap:
        handle.stop()
        return 0
    _serve_until_signalled(handle.stop)
    return 0


def _close_subtensor(subtensor) -> None:
    """Close a Subtensor and its backing interface without double-closing."""
    seen: set[int] = set()
    for target in (
        subtensor,
        getattr(subtensor, "substrate", None),
        getattr(subtensor, "substrate_interface", None),
    ):
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        close = getattr(target, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("failed to close chain connection", exc_info=True)


def _ensure_enc_pubkey_published(
    subtensor,
    wallet,
    netuid: int,
    private_key: bytes,
    pubkey: bytes,
    *,
    owner_hotkey: str,
    network: str | None = None,
    max_attempts: int = 20,
    retry_seconds: float = 3.0,
) -> None:
    from thinker.submission.commitments import (
        commit_validator_keys,
        read_enc_pubkey,
        read_validator_key_grant,
    )
    from thinker.submission.crypto import (
        encrypt_validator_key_for_owner,
        validator_key_grant_fields,
        validator_key_grant_owner_key_id,
    )

    hotkey = wallet.hotkey.ss58_address
    if not owner_hotkey:
        raise RuntimeError(
            "THINKER_OWNER_HOTKEY must be set before validators can share their "
            "encrypted X25519 key grant"
        )
    if owner_hotkey == hotkey:
        raise RuntimeError("owner and validator must use separate hotkeys")
    metagraph_fn = getattr(subtensor, "metagraph", None)
    if callable(metagraph_fn):
        metagraph = metagraph_fn(netuid)
        hotkeys = list(getattr(metagraph, "hotkeys", ()))
        if hotkey not in hotkeys:
            raise RuntimeError(
                f"validator hotkey {hotkey} is not registered on netuid {netuid}"
            )
        permits = getattr(metagraph, "validator_permit", None)
        index = hotkeys.index(hotkey)
        if permits is not None and not bool(permits[index]):
            raise RuntimeError(
                f"validator hotkey {hotkey} has no validator permit on netuid {netuid}; "
                "miners intentionally ignore encryption keys from non-validator hotkeys"
            )

    owner_commitment = read_enc_pubkey(subtensor, netuid, owner_hotkey)
    if owner_commitment is None:
        raise RuntimeError(
            f"owner encryption public key is not published for hotkey={owner_hotkey}, "
            f"netuid={netuid}; the subnet owner must publish its encryption public key first"
        )
    owner_public_key = bytes.fromhex(owner_commitment.pubkey_hex)
    owner_key_id = validator_key_grant_owner_key_id(owner_public_key)

    existing = read_enc_pubkey(subtensor, netuid, hotkey)
    existing_grant = read_validator_key_grant(subtensor, netuid, hotkey)
    if (
        existing is not None
        and existing.pubkey_hex == pubkey.hex()
        and existing_grant is not None
        and existing_grant.owner_key_id_hex == owner_key_id
    ):
        _status(
            f"validator public key and owner grant confirmed on chain: hotkey={hotkey}, "
            f"netuid={netuid}"
        )
        return

    wrapped = encrypt_validator_key_for_owner(private_key, owner_public_key)
    grant_fields = validator_key_grant_fields(wrapped)
    ok, err = commit_validator_keys(
        wallet,
        subtensor,
        netuid,
        pubkey_hex=pubkey.hex(),
        owner_key_id_hex=owner_key_id,
        **grant_fields,
    )
    if not ok:
        raise RuntimeError(f"failed to publish validator key grant: {err}")
    last_seen = None
    for attempt in range(max_attempts):
        if network and attempt > 0 and attempt % 5 == 0:
            # A fresh connection avoids polling the same lagging RPC replica
            # for the whole retry window when the endpoint load-balances.
            try:
                import bittensor as bt

                subtensor = bt.Subtensor(network=network)
            except Exception as exc:
                _status(f"could not reconnect to {network} for readback retry: {exc}")
        published = read_enc_pubkey(subtensor, netuid, hotkey)
        published_grant = read_validator_key_grant(subtensor, netuid, hotkey)
        if (
            published is not None
            and published.pubkey_hex == pubkey.hex()
            and published_grant is not None
            and published_grant.owner_key_id_hex == owner_key_id
        ):
            _status(
                f"validator public key and encrypted owner grant published and verified: hotkey={hotkey}, "
                f"netuid={netuid}"
            )
            return
        last_seen = published
        if attempt == 0:
            _status("validator key grant accepted; waiting for chain readback")
        elif attempt % 5 == 0:
            _status(
                f"still waiting for chain readback (attempt {attempt + 1}/{max_attempts}); "
                f"see logs for the underlying read error if this persists"
            )
        if attempt < max_attempts - 1:
            time.sleep(retry_seconds)
    raise RuntimeError(
        "validator key-grant transaction reported success but it could not "
        f"be read back for hotkey={hotkey}, netuid={netuid} (last readback: {last_seen!r}); "
        "check logs for the underlying chain read error -- this is usually RPC replica lag "
        "on a load-balanced endpoint or a commitment rate limit on the chain"
    )


class OwnerKeyRefreshHandle:
    def __init__(
        self,
        subtensor,
        wallet,
        netuid: int,
        private_key: bytes,
        pubkey: bytes,
        *,
        owner_hotkey: str,
        network: str | None,
        interval_seconds: float = _OWNER_KEY_REFRESH_SECONDS,
        log=_status,
    ) -> None:
        self._subtensor = subtensor
        self._wallet = wallet
        self._netuid = netuid
        self._private_key = private_key
        self._pubkey = pubkey
        self._owner_hotkey = owner_hotkey
        self._network = network
        self._interval_seconds = interval_seconds
        self._log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="owner-key-refresh",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._refresh_once()
            if self._stop.wait(self._interval_seconds):
                break

    def _refresh_once(self) -> None:
        try:
            _ensure_enc_pubkey_published(
                self._subtensor,
                self._wallet,
                self._netuid,
                self._private_key,
                self._pubkey,
                owner_hotkey=self._owner_hotkey,
                network=self._network,
            )
        except Exception as exc:
            logger.warning(
                "owner key refresh skipped; validator grant was not updated",
                exc_info=True,
            )
            self._log(
                "owner key refresh skipped; validator grant was not updated; "
                "will retry on the next refresh: "
                f"{type(exc).__name__}: {exc}"
            )


def _build_inference_backend(args: argparse.Namespace, config: ThinkerConfig):
    from vllm import SamplingParams

    from thinker.validator.inference import BaseModelServer, VllmInferenceBackend

    base_model_path = args.base_model_path or config.base_model_repo
    if not base_model_path:
        raise RuntimeError(
            "no base model configured -- pass --base-model-path or set THINKER_BASE_MODEL_REPO"
        )
    _status(
        f"loading base model for inference: {base_model_path} "
        f"(max_loras={args.max_loras}, max_total_tokens={args.max_total_tokens}, "
        f"max_new_tokens={args.max_new_tokens}, max_num_seqs={args.max_num_seqs}, "
        f"prefix_caching={not args.no_prefix_caching})"
    )
    engine_kwargs = {}
    engine_kwargs["max_model_len"] = args.max_model_len or args.max_total_tokens
    engine_kwargs["max_num_seqs"] = args.max_num_seqs
    engine_kwargs["enable_prefix_caching"] = not args.no_prefix_caching
    server = BaseModelServer(
        base_model_path,
        base_model_revision=config.base_model_revision,
        max_loras=args.max_loras,
        max_lora_rank=config.max_lora_rank,
        security_policy=config,
        show_progress=not args.no_generation_progress,
        **engine_kwargs,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    _status("inference backend ready")
    return VllmInferenceBackend(
        server,
        sampling_params,
        max_total_tokens=args.max_total_tokens,
    )


def _validate_validator_inference_backend(inference) -> None:
    missing = [
        name
        for name in _VALIDATOR_INFERENCE_METHODS
        if not callable(getattr(inference, name, None))
    ]
    if missing:
        raise RuntimeError(
            "validator inference backend is missing required method(s): "
            + ", ".join(missing)
        )


class _OriginalModelSynthesisClient:
    def __init__(self, inference, *, max_new_tokens: int):
        self._inference = inference
        self._max_new_tokens = max(1, int(max_new_tokens))

    def complete(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        _ = (temperature, seed)
        completions = self._inference.generate_original_greedy_limited(
            [prompt],
            max_new_tokens=self._max_new_tokens,
            enable_thinking=False,
        )
        if len(completions) != 1:
            raise ValueError("base model synthesis must return exactly one completion")
        return completions[0][0]


def _register_synthesized_track(config: ThinkerConfig, inference) -> None:
    if not config.synthesized_enabled:
        _status("synthesized math track disabled by THINKER_SYNTHESIZED_ENABLED")
        return
    from thinker.problems.interface import registered_tracks
    from thinker.problems.tracks import synthesized

    if "synthesized" in registered_tracks():
        return
    synthesized.register(
        _OriginalModelSynthesisClient(
            inference,
            max_new_tokens=config.synthesized_max_new_tokens,
        ),
        dataset_name=config.synthesized_dataset,
        split=config.synthesized_split,
        max_scan=config.synthesized_max_scan,
    )
    _status(
        "synthesized math track registered: "
        f"{config.synthesized_dataset}@{config.synthesized_split} "
        "with frozen-base paraphrasing"
    )


class _MockSubmissionTransport:
    def __init__(self, fallback):
        self._fallback = fallback
        self._mock_submission = None

    def set_mock_submission(self, mock_submission) -> None:
        self._mock_submission = mock_submission

    def fetch(self, pointer):
        if pointer.miner_id == _MOCK_MINER_ID:
            if self._mock_submission is None:
                raise ValueError("mock submission has not been selected for this epoch")
            return self._mock_submission
        return self._fallback.fetch(pointer)


def _mock_pointer_from_valid_real_submission(
    pointers: dict,
    *,
    transport,
    config: ThinkerConfig,
    recipient_id: str,
    recipient_privkey: bytes,
):
    from thinker.submission.adapter_validation import validate_adapter_files
    from thinker.submission.crypto import (
        content_hash,
        decrypt_as_recipient,
        max_encrypted_adapter_ciphertext_bytes,
        unpack_adapter_bundle,
    )
    from thinker.validator.epoch_loop import MinerSubmissionPointer

    if not pointers:
        return None, None, None
    candidates = sorted(pointers.items())
    random.shuffle(candidates)
    max_ciphertext = max_encrypted_adapter_ciphertext_bytes(config.max_adapter_bytes)
    rejected: list[str] = []
    for source_miner_id, source_pointer in candidates:
        try:
            submission = transport.fetch(source_pointer)
            if content_hash(submission) != source_pointer.sha256:
                raise ValueError("hash_mismatch")
            if len(submission.ciphertext) > max_ciphertext:
                raise ValueError("encrypted_submission_too_large")
            if len(submission.wrapped_keys) > config.max_submission_recipients:
                raise ValueError("too_many_submission_recipients")
            plaintext = decrypt_as_recipient(
                submission, recipient_id, recipient_privkey
            )
            adapter_files = unpack_adapter_bundle(
                plaintext,
                max_total_bytes=config.max_adapter_bytes,
                max_config_bytes=config.max_adapter_config_bytes,
            )
            validate_adapter_files(adapter_files, config)
        except Exception as exc:
            rejected.append(f"{source_miner_id}: {exc}")
            continue
        return source_miner_id, MinerSubmissionPointer(
            miner_id=_MOCK_MINER_ID,
            epoch=source_pointer.epoch,
            repo_id=source_pointer.repo_id,
            sha256=source_pointer.sha256,
        ), submission
    if rejected:
        _status(
            "mock mode: no discovered submission passed validation "
            f"({'; '.join(rejected[:3])})"
        )
    return None, None, None


def _build_real_transport(args: argparse.Namespace, config: ThinkerConfig):
    from thinker.submission.huggingface import HuggingFaceSubmissionTransport

    return HuggingFaceSubmissionTransport(
        config, token=args.hf_token or os.environ.get("HF_TOKEN")
    )


def _build_wandb_logger(args: argparse.Namespace, config: ThinkerConfig, validator_hotkey: str):
    from thinker.validator.wandb_logger import (
        WandbEpochLogger,
        init_validator_run,
        resolve_wandb_project,
    )

    project_path = args.wandb_project or config.wandb_project
    project, entity = resolve_wandb_project(
        project_path, args.wandb_entity or config.wandb_entity or None
    )
    _status(
        "initializing compulsory online W&B run: "
        f"project={f'{entity}/{project}' if entity else project}"
    )
    wandb_run = init_validator_run(
        project,
        validator_hotkey,
        entity=entity,
    )
    _status("W&B run initialized successfully")
    return WandbEpochLogger(wandb_run)


def _long_context_qa_count(args: argparse.Namespace, config: ThinkerConfig) -> int:
    if args.no_long_context_qa:
        return 0
    if args.long_context_qa_per_epoch is not None:
        return max(0, args.long_context_qa_per_epoch)
    return max(0, config.n_long_context_qa_per_epoch)


def _common_seed_for_epoch(
    repo_id: str,
    *,
    subtensor,
    netuid: int,
    epoch: int,
    recipient_id: str,
    recipient_privkey: bytes,
    owner_hotkey: str,
    token: str | None,
    max_recipients: int,
    download_fn=None,
) -> str | None:
    if not repo_id:
        logger.warning("there is no valid common seed")
        return None
    try:
        from thinker.common_seed import fetch_common_seed
        from thinker.submission.commitments import read_owner_common_seed

        committed = read_owner_common_seed(subtensor, netuid, owner_hotkey)
        if committed is None:
            raise ValueError("owner has no persistent common seed commitment")
        record = fetch_common_seed(
            repo_id,
            seed_commitment_hex=committed.seed_commitment_hex,
            recipient_id=recipient_id,
            recipient_privkey=recipient_privkey,
            owner_hotkey=owner_hotkey,
            token=token,
            max_recipients=max_recipients,
            download_fn=download_fn,
        )
        return record.seed
    except Exception as exc:
        logger.warning("there is no valid common seed")
        logger.debug("common seed rejected for epoch %s: %s", epoch, exc)
        return None


def _miner_pointer_for_uid(subtensor, netuid: int, miner_uid: int, epoch: int):
    from thinker.submission.commitments import read_submission
    from thinker.validator.epoch_loop import MinerSubmissionPointer

    if miner_uid < 0:
        raise ValueError("--miner must be a non-negative UID")
    metagraph = subtensor.metagraph(netuid)
    hotkeys = list(getattr(metagraph, "hotkeys", []))
    if miner_uid >= len(hotkeys):
        raise ValueError(
            f"miner UID {miner_uid} is outside metagraph size {len(hotkeys)}"
        )
    miner_hotkey = hotkeys[miner_uid]
    if not miner_hotkey:
        raise ValueError(f"miner UID {miner_uid} has no hotkey in the metagraph")
    commitment = read_submission(subtensor, netuid, miner_hotkey, epoch)
    if commitment is None:
        raise RuntimeError(
            f"miner UID {miner_uid} has no active submission for epoch {epoch}"
        )
    return miner_hotkey, MinerSubmissionPointer(
        miner_id=miner_hotkey,
        epoch=commitment.epoch,
        repo_id=commitment.repo_id,
        sha256=commitment.sha256,
    )


def _fetch_chat_adapter_files(
    pointer,
    *,
    transport,
    config: ThinkerConfig,
    recipient_id: str,
    recipient_privkey: bytes,
) -> dict[str, bytes]:
    from thinker.submission.adapter_validation import validate_adapter_files
    from thinker.submission.crypto import (
        content_hash,
        decrypt_as_recipient,
        max_encrypted_adapter_ciphertext_bytes,
        unpack_adapter_bundle,
    )

    submission = transport.fetch(pointer)
    if content_hash(submission) != pointer.sha256:
        raise ValueError("downloaded submission hash does not match chain commitment")
    max_ciphertext = max_encrypted_adapter_ciphertext_bytes(config.max_adapter_bytes)
    if len(submission.ciphertext) > max_ciphertext:
        raise ValueError("encrypted submission exceeds validator ciphertext limit")
    if len(submission.wrapped_keys) > config.max_submission_recipients:
        raise ValueError("encrypted submission has too many wrapped recipient keys")
    plaintext = decrypt_as_recipient(submission, recipient_id, recipient_privkey)
    adapter_files = unpack_adapter_bundle(
        plaintext,
        max_total_bytes=config.max_adapter_bytes,
        max_config_bytes=config.max_adapter_config_bytes,
    )
    return validate_adapter_files(adapter_files, config).files


def _chat_messages(
    *,
    system_prompt: str | None,
    history: list[dict[str, str]],
    user_message: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def _run_chat(args: argparse.Namespace) -> int:
    import bittensor as bt

    from thinker.validator import keystore
    from thinker.wallet import load_wallet

    config = load_config()
    inference = None
    subtensor = None
    try:
        _status(
            f"starting miner chat: network={args.network}, netuid={args.netuid}, "
            f"miner_uid={args.miner}, history={args.history}"
        )
        _status(f"connecting to Bittensor network: {args.network}")
        subtensor = bt.Subtensor(network=args.network)
        _status("loading validator encryption key")
        privkey, _pubkey = keystore.load_or_create_keypair(args.key_path)
        try:
            wallet = load_wallet(args.wallet, args.hotkey)
        except Exception as exc:
            _status(
                f"failed to load wallet={args.wallet}, hotkey={args.hotkey}: "
                f"{type(exc).__name__}: {exc}"
            )
            return 1

        current_block = subtensor.get_current_block()
        epoch = current_block // config.epoch_blocks
        miner_hotkey, pointer = _miner_pointer_for_uid(
            subtensor, args.netuid, args.miner, epoch
        )
        _status(
            f"miner UID {args.miner} -> {miner_hotkey}; "
            f"submission_epoch={pointer.epoch}, sha256={pointer.sha256[:12]}..."
        )
        transport = _build_real_transport(args, config)
        adapter_files = _fetch_chat_adapter_files(
            pointer,
            transport=transport,
            config=config,
            recipient_id=wallet.hotkey.ss58_address,
            recipient_privkey=privkey,
        )
        _status("adapter fetched, decrypted, and validated")
        inference = _build_inference_backend(args, config)

        print(
            "Interactive miner chat ready. Type /exit to quit"
            + (" or /clear to clear history." if args.history else "."),
            flush=True,
        )
        history: list[dict[str, str]] = []
        while True:
            try:
                user_message = input("you> ")
            except EOFError:
                print("", flush=True)
                break
            except KeyboardInterrupt:
                print("", flush=True)
                return 130
            stripped = user_message.strip()
            if not stripped:
                continue
            if stripped in {"/exit", "/quit"}:
                break
            if args.history and stripped == "/clear":
                history.clear()
                print("history cleared", flush=True)
                continue
            messages = _chat_messages(
                system_prompt=args.system_prompt,
                history=history if args.history else [],
                user_message=user_message,
            )
            try:
                completion, token_count = inference.generate_chat(
                    miner_hotkey,
                    adapter_files,
                    messages,
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=True,
                )
            except Exception as exc:
                _status(f"chat generation failed: {type(exc).__name__}: {exc}")
                return 1
            print(f"miner> {completion}", flush=True)
            print(f"[tokens generated: {token_count}]", flush=True)
            if args.history:
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": completion})
        return 0
    except Exception as exc:
        logger.exception("miner chat failed")
        _status(f"chat failed with {type(exc).__name__}: {exc}")
        return 1
    finally:
        if inference is not None:
            close_inference = getattr(inference, "close", None)
            if callable(close_inference):
                close_inference()
        _close_subtensor(subtensor)


def _run_validator_epochs(
    *,
    args: argparse.Namespace,
    config: ThinkerConfig,
    evaluation_delay_epochs: int,
    subtensor,
    wallet,
    recipient_privkey: bytes,
    transport,
    real_transport,
    loop,
    wandb_logger,
    long_context_qa_count: int,
) -> int:
    from thinker.validator import chain

    last_epoch: int | None = None
    epochs_run = 0
    common_seed_repo = args.common_seed_repo or config.common_seed_repo
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    while True:
        epoch: int | None = None
        try:
            current_block = subtensor.get_current_block()
            epoch = current_block // config.epoch_blocks
            if epoch == last_epoch:
                time.sleep(args.poll_seconds)
                continue

            _status(f"epoch {epoch}: reading metagraph and miner commitments")
            metagraph = subtensor.metagraph(args.netuid)
            maturing_submissions: list[chain.MaturingSubmission] = []
            pointers = chain.discover_miner_pointers(
                subtensor,
                args.netuid,
                metagraph,
                epoch,
                epoch_blocks=config.epoch_blocks,
                evaluation_delay_epochs=evaluation_delay_epochs,
                current_block=current_block,
                maturing_submissions=maturing_submissions,
            )
            for pending in sorted(
                maturing_submissions, key=lambda item: item.eligible_block
            ):
                _status(
                    f"epoch {epoch}: miner uid={pending.uid} submission matures in "
                    f"{pending.remaining_blocks} block(s), at block "
                    f"{pending.eligible_block} (epoch {pending.eligible_epoch})"
                )

            if args.mock:
                source_miner_id, mock_pointer, mock_submission = (
                    _mock_pointer_from_valid_real_submission(
                        pointers,
                        transport=real_transport,
                        config=config,
                        recipient_id=wallet.hotkey.ss58_address,
                        recipient_privkey=recipient_privkey,
                    )
                )
                if mock_pointer is not None:
                    pointers[_MOCK_MINER_ID] = mock_pointer
                    transport.set_mock_submission(mock_submission)
                    _status(
                        f"epoch {epoch}: mock miner uses submission from "
                        f"{source_miner_id}"
                    )

            if not pointers:
                if maturing_submissions:
                    _status(
                        f"epoch {epoch}: waiting for "
                        f"{len(maturing_submissions)} maturing miner submission(s); "
                        f"retrying discovery in {_EMPTY_ROUND_RETRY_SECONDS}s"
                    )
                else:
                    _status(
                        f"epoch {epoch}: no miner submissions; retrying discovery "
                        f"in {_EMPTY_ROUND_RETRY_SECONDS}s"
                    )
                time.sleep(_EMPTY_ROUND_RETRY_SECONDS)
                # Keep last_epoch unchanged so this epoch is retried instead
                # of being treated as an already-completed empty round.
                continue

            print(
                f"epoch {epoch}: scoring {len(pointers)} miner submission(s)",
                flush=True,
            )
            _status(
                f"epoch {epoch}: running evaluation batch; "
                "this may include model generation"
            )
            common_seed = _common_seed_for_epoch(
                common_seed_repo,
                subtensor=subtensor,
                netuid=args.netuid,
                epoch=epoch,
                recipient_id=wallet.hotkey.ss58_address,
                recipient_privkey=recipient_privkey,
                owner_hotkey=config.owner_hotkey,
                token=hf_token,
                max_recipients=config.max_submission_recipients,
            )
            if wandb_logger is not None:
                try:
                    wandb_logger.start_epoch(epoch, pointers)
                except Exception as exc:
                    logger.warning(
                        "W&B progress initialization failed for epoch %s: %s",
                        epoch,
                        type(exc).__name__,
                    )
            results = loop.run_epoch(
                pointers,
                n_long_context_qa=long_context_qa_count,
                n_multiple_choice=config.qualification_multiple_choice_per_epoch,
                epoch=epoch,
                common_seed=common_seed,
                test_mode=args.test_mode,
            )
            if wandb_logger is not None:
                try:
                    wandb_logger.log_epoch(epoch, results)
                except Exception as exc:
                    logger.exception(
                        "compulsory W&B logging failed for epoch %s", epoch
                    )
                    _status(
                        f"epoch {epoch}: compulsory W&B logging failed with "
                        f"{type(exc).__name__}: {exc}; stopping validator"
                    )
                    return 1
                _status(f"epoch {epoch}: W&B metrics logged")
            elif args.test_mode is not None:
                _status(
                    f"epoch {epoch}: test mode {args.test_mode}; W&B logging skipped"
                )
            for miner_id, result in results.items():
                if result.score is not None:
                    print(
                        f"  {miner_id}: overall={result.score.overall:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {miner_id}: rejected ({result.rejected_reason})",
                        flush=True,
                    )
        except Exception as exc:
            epoch_label = "unknown" if epoch is None else str(epoch)
            if epoch is not None:
                if wandb_logger is not None:
                    try:
                        wandb_logger.fail_epoch(epoch)
                    except Exception:
                        logger.warning(
                            "W&B progress failure status could not be logged for "
                            "epoch %s",
                            epoch,
                        )
            logger.exception("validator epoch %s failed", epoch_label)
            _status(
                f"epoch {epoch_label}: failed with "
                f"{type(exc).__name__}: {exc}"
            )
            if args.once:
                return 1
            _status("epoch failure is recoverable; waiting for the next epoch")
            if epoch is not None:
                last_epoch = epoch
            epochs_run += 1
            if args.max_epochs is not None and epochs_run >= args.max_epochs:
                break
            if epoch is None:
                time.sleep(args.poll_seconds)
            continue

        last_epoch = epoch
        epochs_run += 1
        if args.once or (
            args.max_epochs is not None and epochs_run >= args.max_epochs
        ):
            break
    return 0


def _run_validator_loop(args: argparse.Namespace) -> int:
    import bittensor as bt

    from thinker.problems.decontam import DecontaminationStore
    from thinker.problems.tracks import register_default_tracks
    from thinker.validator import chain, keystore
    from thinker.validator.epoch_loop import EpochLoop
    from thinker.wallet import load_wallet

    config = load_config()
    evaluation_delay_epochs = (
        chain.MODEL_EVALUATION_DELAY_EPOCHS
        if args.evaluation_delay_epochs is None
        else args.evaluation_delay_epochs
    )
    register_default_tracks()
    retrieval_handle = None
    wandb_logger = None
    inference = None
    weight_setter = None
    owner_key_refresher = None
    subtensor = None
    owner_key_subtensor = None
    weight_subtensor = None
    try:
        test_mode = args.test_mode
        long_context_qa_count = _long_context_qa_count(args, config)
        if test_mode in {"math", "science"}:
            long_context_qa_count = 0
        if test_mode == "long_qa" and long_context_qa_count <= 0:
            _status(
                "--test-mode long_qa requires at least one long-context QA sample; "
                "remove --no-long-context-qa or set THINKER_N_LONG_CONTEXT_QA_PER_EPOCH above 0"
            )
            return 1
        if (
            test_mode == "science"
            and config.qualification_multiple_choice_per_epoch <= 0
        ):
            _status(
                "--test-mode science requires THINKER_QUALIFICATION_MULTIPLE_CHOICE_PER_EPOCH above 0"
            )
            return 1
        needs_retrieval = (
            test_mode == "long_qa"
            or (test_mode is None and long_context_qa_count > 0)
        )
        _status(
            f"starting validator run: network={args.network}, netuid={args.netuid}, "
            f"mock={args.mock}, set_weights={not args.no_set_weights and test_mode is None}, "
            f"burn_rate={args.burn_rate}, "
            f"evaluation_delay_epochs={evaluation_delay_epochs}, "
            f"test_mode={test_mode or 'off'}"
        )
        if not args.no_retrieval and needs_retrieval:
            retrieval_handle = start_validator_runtime(config, args)
            print(f"Thinker retrieval service ready at {retrieval_handle.url}/retrieve", flush=True)
        elif not needs_retrieval:
            _status("retrieval bootstrap skipped; selected evaluation mode does not use long-context QA")
        else:
            _status("retrieval bootstrap disabled by --no-retrieval")

        _status(f"connecting to Bittensor network: {args.network}")
        subtensor = bt.Subtensor(network=args.network)
        # substrate-interface WebSocket receives are not thread-safe.  Each
        # background worker owns a separate connection so its periodic RPCs
        # cannot collide with discovery/common-seed reads in the epoch loop.
        owner_key_subtensor = bt.Subtensor(network=args.network)
        weight_subtensor = bt.Subtensor(network=args.network)
        _status("loading validator encryption key")
        privkey, pubkey = keystore.load_or_create_keypair(args.key_path)
        try:
            wallet = load_wallet(args.wallet, args.hotkey)
        except Exception as exc:
            _status(
                f"failed to load wallet={args.wallet}, hotkey={args.hotkey}: "
                f"{type(exc).__name__}: {exc}"
            )
            return 1

        if args.mock:
            print(
                f"mock mode: using wallet hotkey {wallet.hotkey.ss58_address}",
                flush=True,
            )
        _status(f"using wallet={args.wallet}, hotkey={args.hotkey}")
        owner_key_refresher = OwnerKeyRefreshHandle(
            owner_key_subtensor,
            wallet,
            args.netuid,
            privkey,
            pubkey,
            owner_hotkey=config.owner_hotkey,
            network=args.network,
            interval_seconds=_OWNER_KEY_REFRESH_SECONDS,
            log=_status,
        )
        _status(
            "starting owner-key refresh thread: check immediately, then every "
            f"{_OWNER_KEY_REFRESH_SECONDS}s"
        )
        owner_key_refresher.start()
        if args.no_set_weights or test_mode is not None:
            weight_setter = chain.NullWeightSetter(log=_status)
            if test_mode is not None:
                _status(
                    f"test mode {test_mode}: weight-setting disabled; scores will be computed but not written to chain"
                )
            else:
                _status("weight-setting disabled by --no-set-weights; scores will be computed but not written to chain")
        else:
            weight_setter = chain.PeriodicWeightSetter(
                chain.BittensorWeightSetter(
                    weight_subtensor,
                    wallet,
                    args.netuid,
                    burn_rate=args.burn_rate,
                ),
                interval_seconds=_WEIGHT_RETRY_SECONDS,
                log=_status,
            )
            _status(
                "starting background weight-setter thread: waiting for evaluation results, "
                f"then writing the latest scores every {_WEIGHT_RETRY_SECONDS}s"
            )
        weight_setter.start()
        if test_mode is None:
            wandb_logger = _build_wandb_logger(
                args, config, wallet.hotkey.ss58_address
            )
        else:
            _status(f"test mode {test_mode}: W&B logging disabled")

        if needs_retrieval and retrieval_handle is None:
            raise RuntimeError(
                "long-context QA requires the local retrieval service; remove --no-retrieval, "
                "pass --no-long-context-qa, or set THINKER_N_LONG_CONTEXT_QA_PER_EPOCH=0"
            )

        inference = _build_inference_backend(args, config)
        _validate_validator_inference_backend(inference)
        _register_synthesized_track(config, inference)
        if args.mock:
            _status(
                "mock mode: evaluating all discovered miners plus mock-miner-1 "
                "cloned from one random valid submission"
            )

        long_context_evaluator = None
        if long_context_qa_count > 0:
            from thinker.validator.long_context_qa import LongContextQAEvaluator

            long_context_evaluator = LongContextQAEvaluator(
                retriever=retrieval_handle.server.retriever,
                inference=inference,
                show_progress=not args.no_generation_progress,
            )
            _status(
                f"epoch composition: {config.n_problems_per_epoch} math sample(s), "
                f"{long_context_qa_count} long-context QA sample(s)"
            )
        else:
            _status(f"epoch composition: {config.n_problems_per_epoch} math sample(s), 0 long-context QA sample(s)")
        if test_mode is not None:
            _status(
                f"test mode {test_mode}: task-only full evaluation; "
                "qualification, W&B, chain weights, cache, and round-state updates are skipped"
            )
        qualification_thinking = min(
            max(0, config.qualification_multiple_choice_thinking_per_epoch),
            max(0, config.qualification_multiple_choice_per_epoch),
        )
        qualification_no_thinking = max(
            0,
            config.qualification_multiple_choice_per_epoch - qualification_thinking,
        )
        _status(
            "staged evaluation: "
            f"qualification={config.qualification_math_per_epoch} math/"
            f"{config.qualification_long_context_qa_per_epoch} long-context QA/"
            f"{config.qualification_multiple_choice_per_epoch} multiple-choice, "
            f"qualification thinking split="
            f"{qualification_thinking}/{qualification_no_thinking}, "
            f"qualification dataset={config.multiple_choice_dataset}@{config.multiple_choice_split}, "
            f"qualification max_new_tokens={config.multiple_choice_max_new_tokens}, "
            f"full top-k={config.full_eval_top_k}, "
            f"full EMA alpha={config.full_eval_ema_alpha:.2f}, "
            f"champion history={config.champion_history_rounds} round(s), "
            f"skip after={config.full_eval_skip_after_rounds} round(s), "
            f"cache={config.eval_cache_path}, "
            f"round_state={config.round_state_path}"
        )

        real_transport = _build_real_transport(args, config)
        transport = (
            _MockSubmissionTransport(real_transport) if args.mock else real_transport
        )

        _status(f"opening decontamination store: {args.decontam_path}")
        decontam = DecontaminationStore(args.decontam_path)
        from thinker.validator.multiple_choice import MultipleChoiceEvaluator

        multiple_choice_evaluator = MultipleChoiceEvaluator(
            inference=inference,
            dataset_name=config.multiple_choice_dataset,
            split=config.multiple_choice_split,
            max_new_tokens=config.multiple_choice_max_new_tokens,
        )
        _status("epoch loop ready; waiting for first epoch")
        loop = EpochLoop(
            config=config,
            recipient_id=wallet.hotkey.ss58_address,
            recipient_privkey=privkey,
            transport=transport,
            inference=inference,
            weight_setter=weight_setter,
            decontam_store=decontam,
            seed_fn=lambda: os.urandom(16).hex(),
            long_context_evaluator=long_context_evaluator,
            multiple_choice_evaluator=multiple_choice_evaluator,
            fingerprint_exempt_miner_ids=(
                {_MOCK_MINER_ID} if args.mock else None
            ),
            show_progress=not args.no_generation_progress,
            progress_callback=(
                wandb_logger.log_progress if wandb_logger is not None else None
            ),
        )

        return _run_validator_epochs(
            args=args,
            config=config,
            evaluation_delay_epochs=evaluation_delay_epochs,
            subtensor=subtensor,
            wallet=wallet,
            recipient_privkey=privkey,
            transport=transport,
            real_transport=real_transport,
            loop=loop,
            wandb_logger=wandb_logger,
            long_context_qa_count=long_context_qa_count,
        )
    finally:
        from thinker.interrupts import was_interrupted

        interrupted = was_interrupted()
        if owner_key_refresher is not None:
            owner_key_refresher.stop()
        if weight_setter is not None:
            weight_setter.stop()
        for chain_connection in (subtensor, owner_key_subtensor, weight_subtensor):
            _close_subtensor(chain_connection)
        if inference is not None:
            close_inference = getattr(inference, "close", None)
            if callable(close_inference):
                close_inference()
        if wandb_logger is not None:
            if interrupted:
                _status(
                    "interrupt received; skipping blocking W&B finish. "
                    "Run data remains on disk and can be synced later with wandb sync."
                )
            else:
                wandb_logger.close()
        if retrieval_handle is not None:
            retrieval_handle.stop()


def main(argv: list[str] | None = None) -> int:
    from thinker.interrupts import interrupt_was_requested, interruptible_process

    try:
        with interruptible_process("thinker-validator"):
            parser = build_parser()
            args = parser.parse_args(argv)
            if args.command is None:
                parser.print_help()
                return 0
            if args.command == "retrieve":
                return _run_retrieve(args)
            if args.command == "run":
                return _run_validator_loop(args)
            if args.command == "chat":
                return _run_chat(args)
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        if interrupt_was_requested():
            os._exit(130)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
