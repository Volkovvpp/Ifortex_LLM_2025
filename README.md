# Text Summarization with HuggingFace Transformers

This project provides a simple pipeline for summarizing long texts using the `facebook/bart-large-cnn` model from HuggingFace Transformers, with support for chunking large texts and running inference on GPU (CUDA).
This model has several advantages that make it a popular choice for text summarization tasks. It is specially trained on the corpus of news articles and their short extracts, which allows it to effectively extract the essence from long texts. Thanks to its BART architecture, the model combines the capabilities of both context understanding (like BERT) and text generation (like GPT), which ensures high-quality final results. In addition, it is powerful enough to give good results, but it is not too resource-intensive, so it is suitable for use even on ordinary computers.

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

## Usage example 

To demonstrate how the app works I used text "Snow White and The Seven Dwarfs".
Result and the original text we be represented in directory `example`