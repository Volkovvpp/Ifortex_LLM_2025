#Установка переменной окружения для указания папки кэша
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

import gradio as gr
from text_processing.summarize import summarize_with_overlap
from text_processing.file_loader import read_file

THEME = gr.themes.Soft()
TITLE = "Summary"

#Функция для кнопки "Generate"
def process_input(text: str, file: gr.File) -> str:
    try:
        if file:
            text = read_file(file.name)
        if not text.strip():
            return "Error: Input text or load file."
        return summarize_with_overlap(text)
    except Exception as e:
        return f"Error: {str(e)}"

#Создание интерфейса
with gr.Blocks(title=TITLE, theme=THEME) as app:
    with gr.Row():
        #Поле для ввода текста вручную
        text_input = gr.Textbox(label="Text", lines=10, placeholder="Input text...")
        #Поле для загрузки файлов
        file_input = gr.File(label="File", file_types=[".txt"])

    #Кнопка запуска обработки
    submit_btn = gr.Button("Generate", variant="primary")
    #Поле вывода результата суммаризации
    output = gr.Textbox(label="Summary", interactive=False)

    #Привязка нажатия кнопки к функции process_input
    submit_btn.click(
        fn=process_input,
        inputs=[text_input, file_input],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(server_port=7860)