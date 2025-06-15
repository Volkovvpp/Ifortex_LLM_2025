from utils.config import MODEL_NAME, ENV_PATH

import os
from transformers import AutoTokenizer

os.environ['HF_HOME'] = ENV_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chunk_with_overlap(text, max_length=512, overlap=32):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    if total_tokens <= max_length:
        return [tokens]

    step = max_length - overlap
    if step <= 0:
        raise ValueError("Перекрытие (overlap) должно быть меньше max_length")

    chunks = []
    for i in range(0, total_tokens, step):
        chunk_tokens = tokens[i:i + max_length]

        # chunk_text = tokenizer.decode(
        #     chunk_tokens,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True
        # )
        chunks.append(chunk_tokens)

    print(f"[DEBUG] Разбито на {len(chunks)} чанков (из {total_tokens} токенов)")
    return chunks