def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

        if start < 0:
            start = 0
    
    return chunks


def chunked_summarize(text: str,summarize_func, max_chunk_size: int = 2000) -> str:
    # 1.split text into chunks
    text_chunks = chunk_text(text, chunk_size=max_chunk_size, overlap=200)
    
    # 2.summarize each chunk individually
    partial_summaries = [summarize_func(chunk) for chunk in text_chunks]
    
    # 3.combine the partial summaries into a final summary
    combined_summary_input = " ".join(partial_summaries)
    final_summary = summarize_func(combined_summary_input)
    return final_summary