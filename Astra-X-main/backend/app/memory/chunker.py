from __future__ import annotations

from uuid import uuid4

from app.memory.models.chunk import Chunk, ChunkResult


class Chunker:
    """Splits documents into chunks for embedding and retrieval.

    Kept separate from the embedder and retriever so chunking strategy
    can evolve independently.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def chunk_text(
        self,
        text: str,
        document_id: str,
        *,
        strategy: str = "fixed_size",
    ) -> ChunkResult:
        if strategy == "fixed_size":
            return self._fixed_size(text, document_id)
        if strategy == "paragraph":
            return self._paragraph(text, document_id)
        if strategy == "sentence":
            return self._sentence(text, document_id)
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _fixed_size(self, text: str, document_id: str) -> ChunkResult:
        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            content = text[start:end]
            chunks.append(
                Chunk(
                    id=str(uuid4()),
                    document_id=document_id,
                    content=content,
                    chunk_index=index,
                    metadata={"start": start, "end": end},
                )
            )
            index += 1
            start += self._chunk_size - self._chunk_overlap

        return ChunkResult(chunks=chunks, strategy="fixed_size")

    def _paragraph(self, text: str, document_id: str) -> ChunkResult:
        import re
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[Chunk] = []
        for i, para in enumerate(paragraphs):
            stripped = para.strip()
            if not stripped:
                continue
            chunks.append(
                Chunk(
                    id=str(uuid4()),
                    document_id=document_id,
                    content=stripped,
                    chunk_index=i,
                    metadata={"type": "paragraph"},
                )
            )
        return ChunkResult(chunks=chunks, strategy="paragraph")

    def _sentence(self, text: str, document_id: str) -> ChunkResult:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_len = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            buffer.append(sent)
            buffer_len += len(sent)
            if buffer_len >= self._chunk_size:
                chunks.append(
                    Chunk(
                        id=str(uuid4()),
                        document_id=document_id,
                        content=" ".join(buffer),
                        chunk_index=len(chunks),
                        metadata={"type": "sentence_group"},
                    )
                )
                buffer = []
                buffer_len = 0

        if buffer:
            chunks.append(
                Chunk(
                    id=str(uuid4()),
                    document_id=document_id,
                    content=" ".join(buffer),
                    chunk_index=len(chunks),
                    metadata={"type": "sentence_group"},
                )
            )

        return ChunkResult(chunks=chunks, strategy="sentence")
