from __future__ import annotations


def load_wallet(wallet_name: str, hotkey_name: str) -> object:
    from bittensor_wallet import Wallet

    wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
    if not wallet.hotkey_file.exists_on_device():
        raise SystemExit(f"hotkey {hotkey_name!r} not found for wallet {wallet_name!r}")
    return wallet
