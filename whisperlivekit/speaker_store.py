from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import chromadb


class SpeakerStore:
    def __init__(self, path: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _require_source_model(source_model: Optional[str]) -> str:
        if not source_model or not str(source_model).strip():
            raise ValueError("source_model is required for speaker storage")
        return str(source_model)

    @staticmethod
    def _merge_metadata(
        existing: Optional[Dict[str, Any]], new_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(existing or {})
        for key, value in new_values.items():
            if value is not None:
                merged[key] = value
        return merged

    def _get_existing(self, speaker_id: str) -> Dict[str, Any]:
        result = self.collection.get(
            ids=[speaker_id], include=["metadatas", "embeddings"]
        )
        if not result.get("ids"):
            return {}
        metadata = (result.get("metadatas") or [{}])[0] or {}
        embedding = None
        embeddings = result.get("embeddings")
        if embeddings is not None:
            if len(embeddings) > 0:
                embedding = embeddings[0]
        return {"metadata": metadata, "embedding": embedding}

    def upsert_speaker(
        self,
        speaker_id: str,
        embedding: Iterable[float],
        source_model: str,
        name: Optional[str] = None,
        recording_id: Optional[str] = None,
    ) -> None:
        source_model = self._require_source_model(source_model)
        existing = self._get_existing(speaker_id)
        metadata = self._merge_metadata(
            existing.get("metadata"),
            {
                "name": name,
                "recording_id": recording_id,
                "source_model": source_model,
            },
        )
        self.collection.upsert(
            ids=[speaker_id], embeddings=[list(embedding)], metadatas=[metadata]
        )

    def get_speaker(
        self, speaker_id: str, include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        include = ["metadatas"]
        if include_embedding:
            include.append("embeddings")
        result = self.collection.get(ids=[speaker_id], include=include)
        if not result.get("ids"):
            return None
        metadata = (result.get("metadatas") or [{}])[0] or {}
        embedding = None
        if include_embedding:
            embeddings = result.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                embedding = embeddings[0]
        return {
            "id": speaker_id,
            "embedding": embedding,
            "metadata": metadata,
        }

    def update_speaker(
        self,
        speaker_id: str,
        *,
        name: Optional[str] = None,
        source_model: Optional[str] = None,
        recording_id: Optional[str] = None,
        embedding: Optional[Iterable[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = self._get_existing(speaker_id)
        if not existing:
            return None
        if source_model is None:
            source_model = existing.get("metadata", {}).get("source_model")
        source_model = self._require_source_model(source_model)
        metadata = self._merge_metadata(
            existing.get("metadata"),
            {
                "name": name,
                "recording_id": recording_id,
                "source_model": source_model,
            },
        )
        final_embedding = list(embedding) if embedding is not None else existing.get("embedding")
        if final_embedding is None:
            raise ValueError("embedding is required when no existing embedding is stored")
        self.collection.upsert(
            ids=[speaker_id], embeddings=[final_embedding], metadatas=[metadata]
        )
        return {"id": speaker_id, "embedding": final_embedding, "metadata": metadata}

    def delete_speaker(self, speaker_id: str) -> bool:
        existing = self.collection.get(ids=[speaker_id])
        if not existing.get("ids"):
            return False
        self.collection.delete(ids=[speaker_id])
        return True

    def query_candidates(
        self,
        embedding: Iterable[float],
        *,
        topk: int = 3,
        threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        if topk <= 0:
            return []
        try:
            result = self.collection.query(
                query_embeddings=[list(embedding)],
                n_results=topk,
                include=["metadatas", "distances", "ids"],
            )
        except Exception:
            return []
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        candidates: List[Dict[str, Any]] = []
        for spk_id, distance, meta in zip(ids, distances, metadatas):
            if meta is None:
                continue
            name = meta.get("name")
            if not name:
                continue
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue
            candidates.append(
                {
                    "id": spk_id,
                    "name": name,
                    "similarity": similarity,
                }
            )
        return candidates
