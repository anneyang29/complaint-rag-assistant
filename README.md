# Complaint Investigation RAG

A personal RAG project for complaint analysis using historical consumer complaint cases.

## What this project does

- retrieves similar historical complaint cases
- retrieves supporting evidence chunks
- generates investigation-style outputs such as:
  - likely issue category
  - core conflict point
  - similarity pivot
  - risk alert
  - suggested review checks
- provides an interactive Streamlit demo

## Tech Stack

- **Python**
- **Chroma**
- **HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`)**
- **OpenAI API**
- **Streamlit** 

## Dataset

- Kaggle: [Consumer Complaint Database](https://www.kaggle.com/datasets/selener/consumer-complaint-database)


## Run the app

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install torchvision
$env:OPENAI_API_KEY="your_api_key_here"
python -m streamlit run app.py
