from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from thinker.config import ThinkerConfig
from thinker.submission.commitments import commit_submission, read_enc_pubkey
from thinker.submission.crypto import (
    EncryptedSubmission,
    content_hash,
    encrypt_for_recipients,
    pack_adapter_bundle,
)
from thinker.submission.huggingface import (
    HuggingFaceLocator,
    HuggingFaceTransportError,
    upload_encrypted_submission,
)

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


@dataclass(frozen=True)
class ValidatorRecipient:
    uid: int
    hotkey: str
    pubkey_hex: str


def discover_validator_recipients(subtensor: Any, netuid: int) -> list[ValidatorRecipient]:
    metagraph = subtensor.metagraph(netuid)
    permits = getattr(metagraph, "validator_permit", None)
    recipients: list[ValidatorRecipient] = []
    for uid, hotkey in enumerate(getattr(metagraph, "hotkeys", ())):
        if not hotkey:
            continue
        if permits is not None and (uid >= len(permits) or not bool(permits[uid])):
            continue
        commitment = read_enc_pubkey(subtensor, netuid, hotkey)
        if commitment is None:
            continue
        recipients.append(
            ValidatorRecipient(uid=uid, hotkey=hotkey, pubkey_hex=commitment.pubkey_hex)
        )
    return recipients


def build_submission(
    adapter_dir: Path | str, recipient_pubkeys: dict[str, bytes], config: ThinkerConfig
) -> EncryptedSubmission:
    adapter_dir = Path(adapter_dir)
    files = {
        ADAPTER_CONFIG_NAME: (adapter_dir / ADAPTER_CONFIG_NAME).read_bytes(),
        ADAPTER_WEIGHTS_NAME: (adapter_dir / ADAPTER_WEIGHTS_NAME).read_bytes(),
    }
    plaintext = pack_adapter_bundle(
        files,
        max_total_bytes=config.max_adapter_bytes,
        max_config_bytes=config.max_adapter_config_bytes,
    )
    return encrypt_for_recipients(plaintext, recipient_pubkeys)


def submit(
    *,
    wallet: Any,
    subtensor: Any,
    netuid: int,
    epoch: int,
    adapter_dir: Path | str,
    hf_repo_id: str,
    hf_token: str,
    config: ThinkerConfig,
    recipient_pubkeys: Mapping[str, bytes] | None = None,
    api: Any | None = None,
    locator_sink: Callable[[HuggingFaceLocator], None] | None = None,
) -> tuple[bool, str]:
    if recipient_pubkeys is None:
        recipients = discover_validator_recipients(subtensor, netuid)
        recipient_pubkeys = {
            recipient.hotkey: bytes.fromhex(recipient.pubkey_hex) for recipient in recipients
        }
        scanned = len(getattr(subtensor.metagraph(netuid), "hotkeys", ()))
    else:
        recipient_pubkeys = dict(recipient_pubkeys)
        scanned = 0
    if not recipient_pubkeys:
        scanned_text = (
            f"; scanned {scanned} metagraph hotkey(s)" if scanned else ""
        )
        return False, (
            "no validator encryption pubkeys were selected for this subnet"
            f"{scanned_text}. Check that "
            "validator and miner use the same network/netuid and that validator startup "
            "confirmed its encryption pubkey on chain"
        )

    submission = build_submission(adapter_dir, recipient_pubkeys, config)
    try:
        locator = upload_encrypted_submission(
            hf_repo_id, submission, token=hf_token, api=api
        )
    except HuggingFaceTransportError as exc:
        return False, str(exc)
    if locator_sink is not None:
        locator_sink(locator)
    return commit_submission(
        wallet,
        subtensor,
        netuid,
        epoch,
        repo_id=locator.repo_id,
        sha256=content_hash(submission),
    )
