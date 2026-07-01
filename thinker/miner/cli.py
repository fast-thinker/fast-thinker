from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from thinker.config import NETUID, load_config
from thinker.miner.submit import ValidatorRecipient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thinker-miner",
        description="Thinker miner: encrypt and submit a local LoRA adapter to the subnet.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Encrypt a local LoRA adapter and activate it from an epoch onward.",
    )
    submit_parser.add_argument(
        "--adapter-dir", required=True,
        help="Directory containing adapter_config.json and adapter_model.safetensors",
    )
    submit_parser.add_argument(
        "--hf-repo", required=True, help="Hugging Face repo id to upload the encrypted submission to"
    )
    submit_parser.add_argument(
        "--hf-token", default=None, help="Hugging Face write token (default: $HF_TOKEN)"
    )
    submit_parser.add_argument("--wallet", default="default", help="Bittensor wallet name")
    submit_parser.add_argument("--hotkey", default="default", help="Bittensor hotkey name")
    submit_parser.add_argument("--netuid", type=int, default=NETUID, help="Subnet netuid")
    submit_parser.add_argument(
        "--epoch", type=int, default=None,
        help="First active epoch (default: current epoch, derived from chain block height)",
    )
    submit_parser.add_argument(
        "--network", default="finney", help="Bittensor network: finney, test, or local"
    )
    submit_parser.add_argument(
        "--validator-uids",
        default=None,
        help="Validator recipients: all or comma-separated UIDs such as 1,2,3",
    )

    return parser


def _close_subtensor(subtensor: Any) -> None:
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
                pass


def _short(value: str, *, prefix: int = 8, suffix: int = 6) -> str:
    if len(value) <= prefix + suffix + 3:
        return value
    return f"{value[:prefix]}...{value[-suffix:]}"


def _print_validator_recipients(recipients: list[ValidatorRecipient]) -> None:
    print("validators with valid encryption public keys:")
    for recipient in recipients:
        print(
            f"  uid={recipient.uid:<4} hotkey={_short(recipient.hotkey)} "
            f"pubkey={_short(recipient.pubkey_hex)}"
        )


def _parse_validator_uid_selection(
    raw: str | None, recipients: list[ValidatorRecipient]
) -> tuple[list[ValidatorRecipient] | None, str]:
    by_uid = {recipient.uid: recipient for recipient in recipients}
    available = ", ".join(str(uid) for uid in sorted(by_uid)) or "none"
    selection = (raw or "").strip()
    if not selection:
        return None, "validator UID selection is required"
    if selection.lower() == "all":
        return list(recipients), ""

    selected_uids: list[int] = []
    for part in selection.split(","):
        token = part.strip()
        if not token:
            return None, "validator UID selection contains an empty item"
        try:
            uid = int(token)
        except ValueError:
            return None, (
                "validator recipients must be all or comma-separated integer UIDs"
            )
        if uid not in by_uid:
            return None, (
                f"validator UID {uid} has no valid encryption key on chain; "
                f"available UIDs: {available}"
            )
        if uid not in selected_uids:
            selected_uids.append(uid)
    return [by_uid[uid] for uid in selected_uids], ""


def _select_validator_recipients(
    recipients: list[ValidatorRecipient], requested: str | None
) -> tuple[list[ValidatorRecipient] | None, str]:
    if not recipients:
        return None, "no validators with valid encryption public keys were found on chain"
    _print_validator_recipients(recipients)

    if requested is not None or not sys.stdin.isatty():
        selected, err = _parse_validator_uid_selection(requested, recipients)
        if selected is None:
            return None, err
    else:
        while True:
            raw = input("encrypt for validator UIDs [all or comma-separated UIDs]: ")
            selected, err = _parse_validator_uid_selection(raw, recipients)
            if selected is not None:
                break
            print(f"invalid validator selection: {err}", file=sys.stderr)

    print("selected validator recipient UID(s): " + ",".join(str(item.uid) for item in selected))
    return selected, ""


def _run_submit(args: argparse.Namespace) -> int:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("error: a Hugging Face token is required (--hf-token or $HF_TOKEN)", file=sys.stderr)
        return 1

    config = load_config()

    import bittensor as bt

    from thinker.miner.submit import discover_validator_recipients, submit
    from thinker.validator.chain import current_epoch
    from thinker.wallet import load_wallet

    wallet = load_wallet(args.wallet, args.hotkey)

    subtensor = bt.Subtensor(network=args.network)
    try:
        epoch = args.epoch if args.epoch is not None else current_epoch(subtensor, config.epoch_blocks)
        recipients = discover_validator_recipients(subtensor, args.netuid)
        selected, err = _select_validator_recipients(recipients, args.validator_uids)
        if selected is None:
            print(f"submission failed: {err}", file=sys.stderr)
            return 1
        recipient_pubkeys = {
            recipient.hotkey: bytes.fromhex(recipient.pubkey_hex) for recipient in selected
        }

        ok, err = submit(
            wallet=wallet,
            subtensor=subtensor,
            netuid=args.netuid,
            epoch=epoch,
            adapter_dir=Path(args.adapter_dir),
            hf_repo_id=args.hf_repo,
            hf_token=hf_token,
            config=config,
            recipient_pubkeys=recipient_pubkeys,
        )
    finally:
        _close_subtensor(subtensor)
    if ok:
        print(f"submitted adapter starting at epoch {epoch}")
        return 0
    print(f"submission failed: {err}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    from thinker.interrupts import interruptible_process

    try:
        with interruptible_process("thinker-miner"):
            parser = build_parser()
            args = parser.parse_args(argv)
            if args.command == "submit":
                return _run_submit(args)
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
