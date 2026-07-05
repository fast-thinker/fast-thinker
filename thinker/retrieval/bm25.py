from __future__ import annotations

import gzip
import json
import logging
import os
import random
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Any

logger = logging.getLogger(__name__)

S3_WIKI18_CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
S3_WIKI18_CORPUS_FILE = "wiki-18.jsonl.gz"

BM25S_PREBUILT_INDEX_REPO = "fast-thinker/thinker-bm25s-wiki18"

BM25S_INDEX_FILES = (
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "vocab.index.json",
    "params.index.json",
)
DOCUMENTS_FILE = "documents.thinker.jsonl"
INDEX_META_FILE = "thinker_index_meta.json"
CORPUS_ARROW_DIR = ".corpus_arrow"


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    title: str
    text: str
    contents: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    document: CorpusDocument
    score: float
    rank: int


class CorpusFormatError(ValueError):
    pass


class RetrievalDownloadResourceError(RuntimeError):
    pass


def _configure_hf_download_environment() -> None:
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "2")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")


def _is_thread_resource_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "failed to spawn thread" in text
        or "resource temporarily unavailable" in text
        or "wouldblock" in text
    )


def _raise_download_resource_error(filename: str, exc: BaseException) -> None:
    raise RetrievalDownloadResourceError(
        "Hugging Face download failed because this instance refused to create "
        f"download worker threads while fetching {filename!r}. Increase the "
        "container pids/thread limit, reduce other running processes, or run "
        "with HF_XET_NUM_CONCURRENT_RANGE_GETS=1 and RAYON_NUM_THREADS=1. "
        "If the instance still cannot run Hugging Face's Xet downloader, set "
        "HF_HUB_DISABLE_XET=1 as a slower fallback."
    ) from exc


def _split_contents(contents: str) -> tuple[str, str]:
    if "\n" not in contents:
        return contents.strip().strip('"'), ""
    title, text = contents.split("\n", 1)
    return title.strip().strip('"'), text.strip()


def _normalize_document(raw: dict[str, object], index: int) -> CorpusDocument:
    contents = str(raw.get("contents") or "")
    title = str(raw.get("title") or "")
    text = str(raw.get("text") or "")
    if contents and (not title or not text):
        parsed_title, parsed_text = _split_contents(contents)
        title = title or parsed_title
        text = text or parsed_text
    if not contents:
        contents = f"{title}\n{text}".strip()

    doc_id = str(raw.get("id") or raw.get("_id") or raw.get("docid") or index)
    metadata = {
        k: v
        for k, v in raw.items()
        if k not in {"id", "_id", "docid", "title", "text", "contents"}
    }
    return CorpusDocument(
        doc_id=doc_id,
        title=title,
        text=text,
        contents=contents,
        metadata=metadata,
    )


def _document_to_json(document: CorpusDocument) -> dict[str, object]:
    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "text": document.text,
        "contents": document.contents,
        "metadata": document.metadata,
    }


def _document_from_json(raw: dict[str, object]) -> CorpusDocument:
    return CorpusDocument(
        doc_id=str(raw["doc_id"]),
        title=str(raw.get("title") or ""),
        text=str(raw.get("text") or ""),
        contents=str(raw.get("contents") or ""),
        metadata=dict(raw.get("metadata") or {}),
    )


class LazyCorpusDocuments(Sequence[CorpusDocument]):
    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        return self._from_row(self._dataset[int(index)])

    @staticmethod
    def _from_row(row: dict[str, object]) -> CorpusDocument:
        if "doc_id" in row:
            return _document_from_json(row)
        return _normalize_document(row, 0)


def _load_arrow_dataset(path: Path):
    from datasets import load_from_disk

    return load_from_disk(str(path))


def _load_jsonl_dataset(path: Path):
    from datasets import load_dataset

    kwargs: dict[str, object] = {}
    if os.name != "nt":
        kwargs["num_proc"] = min(os.cpu_count() or 4, 16)
    return load_dataset(
        "json",
        data_files=str(path),
        split="train",
        **kwargs,
    )


def _load_saved_documents(index_dir: Path) -> Sequence[CorpusDocument]:
    arrow_dir = index_dir / CORPUS_ARROW_DIR
    if arrow_dir.is_dir():
        return LazyCorpusDocuments(_load_arrow_dataset(arrow_dir))

    documents_path = index_dir / DOCUMENTS_FILE
    dataset = _load_jsonl_dataset(documents_path)
    dataset.save_to_disk(str(arrow_dir))
    return LazyCorpusDocuments(dataset)


def _iter_jsonl_lines(corpus_path: Path):
    try:
        if tarfile.is_tarfile(corpus_path):
            with tarfile.open(corpus_path, "r:*") as archive:
                members = [
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith(".jsonl")
                ]
                if not members:
                    raise CorpusFormatError(f"{corpus_path} archive contains no JSONL member")
                member = members[0]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise CorpusFormatError(f"{corpus_path} could not open {member.name}")
                with extracted:
                    for line in extracted:
                        yield line
            return
    except CorpusFormatError:
        raise
    except tarfile.TarError:
        pass

    opener = gzip.open if corpus_path.suffix == ".gz" else open
    with opener(corpus_path, "rb") as handle:
        for line in handle:
            yield line


def load_s3_corpus_jsonl(path: str | Path, *, limit: int | None = None) -> list[CorpusDocument]:
    corpus_path = Path(path)
    documents: list[CorpusDocument] = []
    decode_errors = 0
    try:
        for index, raw_line in enumerate(_iter_jsonl_lines(corpus_path)):
            if limit is not None and len(documents) >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                decode_errors += 1
                text = line.decode("utf-8", errors="replace")
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                preview = text[:120].replace("\n", "\\n").replace("\r", "\\r")
                raise CorpusFormatError(
                    f"{corpus_path} line {index + 1} is not JSONL: {preview!r}"
                ) from exc
            if not isinstance(row, dict):
                raise CorpusFormatError(f"{corpus_path} line {index + 1} is not a JSON object")
            documents.append(_normalize_document(row, index))
    except CorpusFormatError:
        raise
    except (OSError, EOFError) as exc:
        raise CorpusFormatError(f"{corpus_path} could not be read as JSONL: {exc}") from exc
    if decode_errors:
        logger.warning(
            "corpus %s: replaced invalid UTF-8 bytes on %d line(s)", corpus_path, decode_errors
        )
    if not documents:
        raise CorpusFormatError(f"{corpus_path} contains no JSONL documents")
    return documents


def download_s3_wiki18_corpus(
    cache_dir: str | Path,
    *,
    force_download: bool = False,
    repo_id: str | None = None,
) -> Path:
    repo_id = repo_id or S3_WIKI18_CORPUS_REPO
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _hf_hub_download_dataset_file(
        repo_id=repo_id,
        filename=S3_WIKI18_CORPUS_FILE,
        local_dir=cache_dir,
        force_download=force_download,
    )


def _hf_hub_download_dataset_file(
    *,
    repo_id: str,
    filename: str,
    local_dir: Path,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    _configure_hf_download_environment()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface-hub to download retrieval assets") from exc

    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=local_dir,
                force_download=force_download,
            )
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if _is_thread_resource_error(exc):
            _raise_download_resource_error(filename, exc)
        raise


def download_prebuilt_bm25_index(
    index_dir: str | Path,
    *,
    repo_id: str = BM25S_PREBUILT_INDEX_REPO,
    revision: str | None = None,
) -> Path:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    for filename in (*BM25S_INDEX_FILES, DOCUMENTS_FILE):
        _hf_hub_download_dataset_file(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=index_dir,
        )
    return index_dir


class BM25RetrievalService:
    def __init__(
        self,
        documents: Iterable[CorpusDocument] | Sequence[CorpusDocument],
        *,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: str | list[str] | None = "en",
    ):
        self.documents = documents if isinstance(documents, Sequence) else list(documents)
        if not self.documents:
            raise ValueError("BM25RetrievalService requires at least one document")
        self.method = method
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self._build_index()

    @classmethod
    def from_s3_corpus(
        cls,
        corpus_path: str | Path,
        *,
        limit: int | None = None,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: str | list[str] | None = "en",
    ) -> BM25RetrievalService:
        return cls(
            load_s3_corpus_jsonl(corpus_path, limit=limit),
            method=method,
            k1=k1,
            b=b,
            stopwords=stopwords,
        )

    @classmethod
    def build_or_load(
        cls,
        *,
        corpus_path: str | Path,
        index_dir: str | Path,
        limit: int | None = None,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: str | list[str] | None = "en",
        mmap: bool = True,
        force_rebuild: bool = False,
    ) -> BM25RetrievalService:
        index_dir = Path(index_dir)
        if not force_rebuild and cls.index_exists(index_dir):
            return cls.from_saved_index(
                index_dir,
                stopwords=stopwords,
                mmap=mmap,
            )
        service = cls.from_s3_corpus(
            corpus_path,
            limit=limit,
            method=method,
            k1=k1,
            b=b,
            stopwords=stopwords,
        )
        service.save(index_dir, corpus_path=corpus_path, limit=limit)
        return service

    @classmethod
    def from_saved_index(
        cls,
        index_dir: str | Path,
        *,
        stopwords: str | list[str] | None = "en",
        mmap: bool = True,
    ) -> BM25RetrievalService:
        try:
            import bm25s
        except ImportError as exc:
            raise ImportError("Install bm25s to use BM25RetrievalService") from exc

        index_dir = Path(index_dir)
        documents = _load_saved_documents(index_dir)
        retriever = bm25s.BM25.load(index_dir, load_corpus=False, mmap=mmap)
        service = cls.__new__(cls)
        service.documents = documents
        service.method = getattr(retriever, "method", "lucene")
        service.k1 = getattr(retriever, "k1", 1.5)
        service.b = getattr(retriever, "b", 0.75)
        service.stopwords = stopwords
        service._bm25s = bm25s
        service._retriever = retriever
        return service

    @staticmethod
    def index_exists(index_dir: str | Path) -> bool:
        index_dir = Path(index_dir)
        return (
            (index_dir / DOCUMENTS_FILE).is_file()
            and all((index_dir / name).is_file() for name in BM25S_INDEX_FILES)
        )

    def _build_index(self) -> None:
        try:
            import bm25s
        except ImportError as exc:
            raise ImportError("Install bm25s to use BM25RetrievalService") from exc

        self._bm25s = bm25s
        corpus_text = [document.contents for document in self.documents]
        corpus_tokens = bm25s.tokenize(corpus_text, stopwords=self.stopwords)
        self._retriever = bm25s.BM25(
            method=self.method,
            k1=self.k1,
            b=self.b,
        )
        self._retriever.index(corpus_tokens)

    def save(
        self,
        index_dir: str | Path,
        *,
        corpus_path: str | Path | None = None,
        limit: int | None = None,
    ) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / DOCUMENTS_FILE).open("w", encoding="utf-8") as handle:
            for document in self.documents:
                handle.write(json.dumps(_document_to_json(document), ensure_ascii=False) + "\n")
        dataset = _load_jsonl_dataset(index_dir / DOCUMENTS_FILE)
        dataset.save_to_disk(str(index_dir / CORPUS_ARROW_DIR))
        meta = {
            "source": "s3_wiki18",
            "corpus_path": str(corpus_path) if corpus_path is not None else None,
            "limit": limit,
            "num_documents": len(self.documents),
            "method": self.method,
            "k1": self.k1,
            "b": self.b,
        }
        (index_dir / INDEX_META_FILE).write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._retriever.save(index_dir)

    def random_document(self, seed: str) -> CorpusDocument:
        rng = random.Random(seed)
        return self.documents[rng.randrange(len(self.documents))]

    def search(self, query: str, *, topk: int = 10) -> list[RetrievalHit]:
        return self.search_batch([query], topk=topk)[0]

    def search_batch(
        self,
        queries: Sequence[str],
        *,
        topk: int = 10,
    ) -> list[list[RetrievalHit]]:
        """Retrieve several queries in one bm25s call, preserving input order."""
        if topk <= 0:
            raise ValueError("topk must be greater than zero")

        results: list[list[RetrievalHit]] = [[] for _ in queries]
        nonempty = [(index, query) for index, query in enumerate(queries) if query.strip()]
        if not nonempty:
            return results

        topk = min(topk, len(self.documents))
        query_tokens = self._bm25s.tokenize(
            [query for _, query in nonempty],
            stopwords=self.stopwords,
        )
        doc_indexes, scores = self._retriever.retrieve(query_tokens, k=topk)
        for row, (result_index, _) in enumerate(nonempty):
            results[result_index] = [
                RetrievalHit(
                    document=self.documents[int(doc_index)], score=float(score), rank=rank
                )
                for rank, (doc_index, score) in enumerate(
                    zip(doc_indexes[row], scores[row]), start=1
                )
            ]
        return results


def format_hits(hits: Iterable[RetrievalHit], *, max_chars_per_doc: int | None = None) -> str:
    parts: list[str] = []
    for display_index, hit in enumerate(hits, start=1):
        text = hit.document.text or hit.document.contents
        if max_chars_per_doc is not None and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip()
        parts.append(f"Doc {display_index} (Title: {hit.document.title}) {text}")
    return "\n".join(parts)


__all__ = [
    "BM25RetrievalService",
    "BM25S_INDEX_FILES",
    "BM25S_PREBUILT_INDEX_REPO",
    "CorpusDocument",
    "CorpusFormatError",
    "DOCUMENTS_FILE",
    "INDEX_META_FILE",
    "RetrievalHit",
    "RetrievalDownloadResourceError",
    "S3_WIKI18_CORPUS_FILE",
    "S3_WIKI18_CORPUS_REPO",
    "download_prebuilt_bm25_index",
    "download_s3_wiki18_corpus",
    "format_hits",
    "load_s3_corpus_jsonl",
]
