"""Chunked GLiNER inference with source-aligned entity selection."""


def _iter_chunks(text: str, model, chunk_words: int, chunk_overlap: int):
    tokens = list(model.data_processor.words_splitter(text))
    limit = min(chunk_words, int(model.config.max_len))
    if not 0 <= chunk_overlap < limit:
        raise ValueError("chunk_overlap must be smaller than the model's chunk limit")
    for first in range(0, len(tokens), limit - chunk_overlap):
        last = min(first + limit, len(tokens))
        start = 0 if first == 0 else tokens[first][1]
        end = len(text) if last == len(tokens) else tokens[last - 1][2]
        yield start, text[start:end]
        if last == len(tokens):
            break


def extract_entities(
    text: str,
    *,
    model,
    labels: list[str],
    threshold: float = 0.5,
    chunk_words: int = 200,
    chunk_overlap: int = 40,
) -> list[dict]:
    """Extract nonoverlapping mentions using a caller-loaded GLiNER model.

    Loading, device selection, and evaluation mode belong to the caller. Offsets
    refer to the original source; exact duplicates retain the highest score.
    """
    import torch

    unique = {}
    for base, chunk in _iter_chunks(text, model, chunk_words, chunk_overlap):
        with torch.inference_mode():
            predictions = model.predict_entities(chunk, labels, threshold=threshold)
        for prediction in predictions:
            start = base + int(prediction["start"])
            end = base + int(prediction["end"])
            assert 0 <= start < end <= len(text)
            assert text[start:end] == prediction["text"]
            entity = {
                "text": text[start:end],
                "label": prediction["label"],
                "score": float(prediction["score"]),
                "start": start,
                "end": end,
            }
            key = (start, end, entity["label"])
            if key not in unique or entity["score"] > unique[key]["score"]:
                unique[key] = entity
    selected = []
    for entity in sorted(
        unique.values(), key=lambda e: (-e["score"], e["start"], e["end"])
    ):
        if not any(
            entity["start"] < other["end"] and other["start"] < entity["end"]
            for other in selected
        ):
            selected.append(entity)
    return sorted(selected, key=lambda e: (e["start"], e["end"]))
