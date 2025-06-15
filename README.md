# Text Summarization with HuggingFace Transformers

This project provides a simple pipeline for summarizing long texts using the `facebook/bart-large-cnn` model from HuggingFace Transformers, with support for chunking large texts and running inference on GPU (CUDA).

## Features

- Automatic text chunking for long inputs
- Summarization using `facebook/bart-large-cnn`
- GPU acceleration with PyTorch and CUDA 11.8
- Ready for integration with Gradio interface


## How to run application

1. **Clone the repository**:

```bash
git clone https://github.com/Volkovvpp/Ifortex_LLM_2025.git
cd Ifortex_LLM_2025
```

2. **Install dependencies used for the application:**

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

3. **Configure your environment in `config.py` file if you need to.**

4. **Run application:**

```bash
python app.py
```

The application will be run on the port: 7860

Use the next link to go to the application page: http://localhost:7860/.