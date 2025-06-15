import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

import gradio as gr
from text_processing.summarize import summarize_with_overlap
from text_processing.file_loader import read_file

THEME = gr.themes.Soft()
TITLE = "Summary"

def process_input(text: str, file: gr.File) -> str:
    try:
        if file:
            text = read_file(file.name)
        if not text.strip():
            return "Error: Input text or load file."
        return summarize_with_overlap(text)
    except Exception as e:
        return f"Error: {str(e)}"



with gr.Blocks(title=TITLE, theme=THEME) as app:
    with gr.Row():
        text_input = gr.Textbox(label="Text", lines=10, placeholder="Input text...")
        file_input = gr.File(label="File", file_types=[".txt"])

    submit_btn = gr.Button("Generate", variant="primary")
    output = gr.Textbox(label="Summary", interactive=False)

    submit_btn.click(
        fn=process_input,
        inputs=[text_input, file_input],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(server_port=7860)