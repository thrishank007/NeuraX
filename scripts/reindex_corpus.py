"""Re-ingest existing uploads through the current chunker/embedding path.

Re-chunks the indexed source files, embeds via the configured provider
(NIM or MiniLM), and replaces their chunks in the vector store.  Use after
changing chunking logic so deterministic chunk IDs stay in sync:

    venv\\Scripts\\python.exe scripts\\reindex_corpus.py                 # all indexed files
    venv\\Scripts\\python.exe scripts\\reindex_corpus.py docs\\a.pdf    # explicit subset

Paths are passed to ingestion verbatim: the delete-and-replace and the
deterministic chunk IDs both key on the exact string, so relativizing or
absolutizing them would orphan existing chunks and duplicate the document.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from chromadb import PersistentClient

from config import NIM_EMBEDDING_CONFIG
from backend.services.component_registry import ComponentRegistry
from backend.services.document_service import _embed_and_store


def indexed_source_files() -> list[str]:
    client = PersistentClient(path=str(PROJECT_ROOT / "vector_db"))
    col = client.get_collection(NIM_EMBEDDING_CONFIG["collection_name"])
    data = col.get(include=["metadatas"])
    paths = sorted({m["file_path"] for m in data["metadatas"] if m.get("file_path")})
    return [p for p in paths if (PROJECT_ROOT / p).exists()]


def main() -> None:
    targets = sys.argv[1:] or indexed_source_files()
    if not targets:
        print("Nothing to reindex: no explicit paths and collection is empty.")
        return

    registry = ComponentRegistry()
    ingestion = registry.ensure_ingestion()
    registry.ensure_vector_stack()

    for path in targets:
        result = ingestion.process_file(path)
        if not result:
            print(f"SKIP {Path(path).name}: processing failed")
            continue
        first_id = _embed_and_store(registry, result)
        status = "ok" if first_id else "no chunks"
        print(f"{status:10s} {Path(path).name}")


if __name__ == "__main__":
    main()
