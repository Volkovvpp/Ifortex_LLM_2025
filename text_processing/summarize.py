from utils.config import MODEL_NAME, ENV_PATH

import os
os.environ['HF_HOME'] = ENV_PATH

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_processing.preprocessing import chunk_with_overlap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

def summarize_with_overlap(text, max_length=512, overlap=32):
    chunks_in_tokens = chunk_with_overlap(text, max_length, overlap)

    summaries = []

    for chunk in chunks_in_tokens:
        #Тензор с входными токенами на устройстве (CPU/GPU)
        input_ids = torch.tensor([chunk], device=DEVICE)

        #Создание маски внимания
        attention_mask = torch.ones_like(input_ids)

        #Генерация саммари
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            min_length=30,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )

        #Декодирование токенов в текст
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)
