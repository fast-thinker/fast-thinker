from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from thinker.config import ThinkerConfig
from thinker.submission.crypto import (
    EncryptedSubmission,
    content_hash,
    max_encrypted_adapter_ciphertext_bytes,
    max_submission_json_bytes,
    submission_from_json,
    submission_to_json,
)

_REPO_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_PINNED_REVISION_RE = re.compile(r"[0-9a-fA-F]{40,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# Must match commit_submission's chain commitment exactly: the chain
# commitment only carries repo_id + sha256 (see commitments.py for why), so
# both upload and fetch derive the filename from the same fixed prefix +
# content hash instead of transmitting it.
SUBMISSION_FILENAME_PREFIX = "encrypted-submissions"


class HuggingFaceTransportError(ValueError):
    pass


def _huggingface_upload_error_message(
    exc: Exception, *, repo_id: str, repo_type: str
) -> str | None:
    names = {cls.__name__ for cls in type(exc).__mro__}
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if "RepositoryNotFoundError" in names or status_code == 404:
        return (
            f"Hugging Face {repo_type} repo '{repo_id}' was not found or is not accessible. "
            "Create the repo first, pass the exact repo id with --hf-repo, and use a write "
            "token that can access it"
        )
    if status_code in {401, 403}:
        return (
            f"Hugging Face denied upload access to {repo_type} repo '{repo_id}'. "
            "Use a write token with access to that repo"
        )
    if "HfHubHTTPError" in names or "HTTPStatusError" in names:
        return f"Hugging Face upload failed for {repo_type} repo '{repo_id}': {exc}"
    return None


def _is_missing_or_inaccessible_repo_error(exc: Exception) -> bool:
    names = {cls.__name__ for cls in type(exc).__mro__}
    response = getattr(exc, "response", None)
    return "RepositoryNotFoundError" in names or getattr(response, "status_code", None) == 404


def _create_public_repo(api: Any, *, repo_id: str, repo_type: str, token: str) -> None:
    create_repo = getattr(api, "create_repo", None)
    if create_repo is None:
        raise HuggingFaceTransportError(
            f"Hugging Face {repo_type} repo '{repo_id}' was not found or is not accessible, "
            "and this client cannot create repositories automatically"
        )
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False,
            exist_ok=True,
            token=token,
        )
    except Exception as exc:
        message = _huggingface_upload_error_message(exc, repo_id=repo_id, repo_type=repo_type)
        if message is None:
            message = f"Hugging Face could not create public {repo_type} repo '{repo_id}': {exc}"
        raise HuggingFaceTransportError(message) from exc


def _make_repo_public(api: Any, *, repo_id: str, repo_type: str, token: str) -> None:
    update_repo_visibility = getattr(api, "update_repo_visibility", None)
    if update_repo_visibility is None:
        return
    try:
        update_repo_visibility(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False,
            token=token,
        )
    except Exception as exc:
        message = _huggingface_upload_error_message(exc, repo_id=repo_id, repo_type=repo_type)
        if message is None:
            message = f"Hugging Face could not make {repo_type} repo '{repo_id}' public: {exc}"
        raise HuggingFaceTransportError(message) from exc


@dataclass(frozen=True)
class HuggingFaceLocator:
    """Where one encrypted submission blob lives on Hugging Face -- the
    only submission transport this subnet supports. Committed directly on
    chain (thinker/submission/commitments.py's SubmissionCommitment) with
    no resolver indirection: the repo id is public chain metadata, while
    only the encrypted contents are secret (thinker/submission/crypto.py)."""

    repo_id: str
    filename: str
    revision: str | None = None
    repo_type: str = "model"

    def validate(self) -> None:
        if not isinstance(self.repo_id, str):
            raise HuggingFaceTransportError("invalid Hugging Face repo_id")
        parts = self.repo_id.split("/")
        if len(parts) != 2 or any(_REPO_COMPONENT_RE.fullmatch(part) is None for part in parts):
            raise HuggingFaceTransportError("invalid Hugging Face repo_id")
        if not isinstance(self.filename, str):
            raise HuggingFaceTransportError("invalid Hugging Face filename")
        path = PurePosixPath(self.filename)
        if (
            not self.filename
            or len(self.filename) > 512
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.filename
            or any(ord(ch) < 32 for ch in self.filename)
        ):
            raise HuggingFaceTransportError("invalid Hugging Face filename")
        if self.revision is not None and (
            not isinstance(self.revision, str) or _PINNED_REVISION_RE.fullmatch(self.revision) is None
        ):
            raise HuggingFaceTransportError("Hugging Face revision must be an immutable commit hash")
        if not isinstance(self.repo_type, str) or self.repo_type not in {"model", "dataset"}:
            raise HuggingFaceTransportError("repo_type must be model or dataset")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "HuggingFaceLocator":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HuggingFaceTransportError(f"invalid locator JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise HuggingFaceTransportError("locator JSON must be an object")
        allowed = {"repo_id", "filename", "revision", "repo_type"}
        if not set(data) <= allowed:
            raise HuggingFaceTransportError("locator JSON has unexpected fields")
        locator = cls(
            repo_id=data.get("repo_id"),
            filename=data.get("filename"),
            revision=data.get("revision"),
            repo_type=data.get("repo_type", "model"),
        )
        locator.validate()
        return locator


def upload_encrypted_submission(
    repo_id: str,
    submission: EncryptedSubmission,
    *,
    token: str,
    repo_type: str = "model",
    path_prefix: str = SUBMISSION_FILENAME_PREFIX,
    api: Any | None = None,
) -> HuggingFaceLocator:
    """Upload ciphertext only; adapter plaintext never reaches HF.

    Only repo_id + content_hash(submission) are committed on chain
    (commitments.py) -- filename is derived from the hash at fetch time, and
    the returned `revision` here is informational only (the immutable HF
    commit this upload produced), not published. The post-download hash
    check in the validator's fetch path is what actually prevents a
    repository owner or attacker from serving a swapped blob, regardless of
    which revision answers the fetch."""
    if not isinstance(token, str) or not token:
        raise HuggingFaceTransportError("a scoped Hugging Face upload token is required")
    digest = content_hash(submission)
    prefix = str(PurePosixPath(path_prefix))
    if PurePosixPath(prefix).is_absolute() or ".." in PurePosixPath(prefix).parts:
        raise HuggingFaceTransportError("invalid upload path prefix")
    filename = f"{prefix}/{digest}.json" if prefix not in ("", ".") else f"{digest}.json"
    provisional = HuggingFaceLocator(repo_id, filename, "0" * 40, repo_type)
    provisional.validate()

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    raw = submission_to_json(submission).encode("utf-8")
    def _upload_file() -> Any:
        return api.upload_file(
            path_or_fileobj=raw,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload encrypted Thinker submission {digest[:12]}",
            token=token,
        )

    try:
        result = _upload_file()
    except Exception as exc:
        if not _is_missing_or_inaccessible_repo_error(exc):
            message = _huggingface_upload_error_message(exc, repo_id=repo_id, repo_type=repo_type)
            if message is not None:
                raise HuggingFaceTransportError(message) from exc
            raise
        _create_public_repo(api, repo_id=repo_id, repo_type=repo_type, token=token)
        try:
            result = _upload_file()
        except Exception as retry_exc:
            message = _huggingface_upload_error_message(
                retry_exc, repo_id=repo_id, repo_type=repo_type
            )
            if message is not None:
                raise HuggingFaceTransportError(message) from retry_exc
            raise
    _make_repo_public(api, repo_id=repo_id, repo_type=repo_type, token=token)
    revision = getattr(result, "oid", None)
    if not isinstance(revision, str):
        raise HuggingFaceTransportError("Hugging Face did not return an immutable commit id")
    locator = HuggingFaceLocator(repo_id, filename, revision, repo_type)
    locator.validate()
    return locator


def _default_resolve_revision(*, repo_id: str, repo_type: str, token: str | None) -> str | None:
    from huggingface_hub import HfApi

    info = HfApi(token=token).repo_info(repo_id, repo_type=repo_type)
    sha = getattr(info, "sha", None)
    return sha if isinstance(sha, str) else None


def _default_remote_size(
    *, repo_id: str, filename: str, revision: str, repo_type: str, token: str | None
) -> int | None:
    from huggingface_hub import HfApi

    infos = HfApi(token=token).get_paths_info(
        repo_id, [filename], revision=revision, repo_type=repo_type
    )
    if not infos:
        return None
    return getattr(infos[0], "size", None)


class HuggingFaceSubmissionTransport:
    """Fetches the encrypted blob a chain SubmissionCommitment points to,
    without ever invoking model-loading APIs. The pointer must carry
    repo_id + sha256 (thinker/validator/epoch_loop.py's
    MinerSubmissionPointer) -- there is no resolver step, since Hugging
    Face is the only transport this subnet supports. filename is derived
    from sha256 (SUBMISSION_FILENAME_PREFIX/{sha256}.json, matching what
    upload_encrypted_submission writes) rather than transmitted -- see
    commitments.py for why both are kept off chain. The fetch resolves
    HEAD to one immutable commit SHA and reuses it for both the
    pre-download size check and the download itself, so a repository
    owner can't swap in a larger blob between the two calls (a HEAD-to-HEAD
    TOCTOU that would otherwise let the post-download size check run only
    after the oversized file was already pulled in full)."""

    def __init__(
        self,
        policy: ThinkerConfig,
        *,
        token: str | None = None,
        cache_dir: str | None = None,
        download_fn: Callable[..., str] | None = None,
        remote_size_fn: Callable[..., int | None] | None = None,
        resolve_revision_fn: Callable[..., str | None] | None = None,
    ):
        self._policy = policy
        self._token = token
        self._cache_dir = cache_dir
        self._download_fn = download_fn
        self._remote_size_fn = remote_size_fn or _default_remote_size
        self._resolve_revision_fn = resolve_revision_fn or _default_resolve_revision

    def fetch(self, pointer: Any) -> EncryptedSubmission:
        sha256 = pointer.sha256
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise HuggingFaceTransportError("invalid sha256")
        filename = f"{SUBMISSION_FILENAME_PREFIX}/{sha256}.json"
        unpinned_locator = HuggingFaceLocator(pointer.repo_id, filename, None)
        unpinned_locator.validate()
        revision = self._resolve_revision_fn(
            repo_id=unpinned_locator.repo_id,
            repo_type=unpinned_locator.repo_type,
            token=self._token,
        )
        if not isinstance(revision, str):
            raise HuggingFaceTransportError("could not resolve Hugging Face repo revision")
        locator = HuggingFaceLocator(
            unpinned_locator.repo_id, unpinned_locator.filename, revision, unpinned_locator.repo_type
        )
        locator.validate()
        max_json = max_submission_json_bytes(
            self._policy.max_adapter_bytes, self._policy.max_submission_recipients
        )
        remote_size = self._remote_size_fn(
            repo_id=locator.repo_id,
            filename=locator.filename,
            revision=locator.revision,
            repo_type=locator.repo_type,
            token=self._token,
        )
        if remote_size is None or remote_size > max_json:
            raise HuggingFaceTransportError("remote encrypted submission is missing or oversized")

        download_fn = self._download_fn
        if download_fn is None:
            from huggingface_hub import hf_hub_download

            download_fn = hf_hub_download
        local_path = download_fn(
            repo_id=locator.repo_id,
            filename=locator.filename,
            revision=locator.revision,
            repo_type=locator.repo_type,
            token=self._token,
            cache_dir=self._cache_dir,
        )
        path = Path(local_path)
        if not path.is_file() or path.stat().st_size > max_json:
            raise HuggingFaceTransportError("downloaded encrypted submission is missing or oversized")
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise HuggingFaceTransportError("downloaded submission is not UTF-8 JSON") from exc
        try:
            return submission_from_json(
                raw,
                max_ciphertext_bytes=max_encrypted_adapter_ciphertext_bytes(
                    self._policy.max_adapter_bytes
                ),
                max_recipients=self._policy.max_submission_recipients,
            )
        except ValueError as exc:
            raise HuggingFaceTransportError(f"invalid encrypted submission: {exc}") from exc
