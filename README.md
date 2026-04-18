# Research Paper RAG

This project has two main Python files:

- `collect_papers_to_chroma.py`: collects research papers, extracts text, chunks it, and stores embeddings in a local Chroma database.
- `main.py`: starts the Streamlit chat app so you can ask questions over the indexed papers.

Run them in this order:

1. Build or refresh the Chroma database with `collect_papers_to_chroma.py`
2. Start the app with `main.py`

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) recommended for dependency management
- [Ollama](https://ollama.com/) installed and running locally
- An Ollama model pulled locally, for example:

```bash
ollama pull llama3.2:3b
```

## Install dependencies

From the project root:

```bash
uv sync
```

If you prefer running without syncing into the local virtual environment each time, you can also use `uv run ...` directly for the commands below.

## File 1: Run the data collection and indexing script

This script downloads papers from arXiv, extracts PDF text, chunks it, and writes embeddings to the local Chroma DB.

Basic command:

```bash
uv run python collect_papers_to_chroma.py
```

Useful example with explicit options:

```bash
uv run python collect_papers_to_chroma.py \
  --max-papers 100 \
  --start-year 2020 \
  --end-year 2026 \
  --output-dir ./paper_dataset \
  --chroma-dir ./chroma_db \
  --collection-name research_papers_rag
```

What it creates:

- `paper_dataset/pdfs/`: downloaded PDFs
- `paper_dataset/raw_text/`: extracted and cleaned text
- `paper_dataset/paper_metadata.json`: saved paper metadata
- `paper_dataset/build_summary.json`: summary of the indexing run
- `chroma_db/`: persistent Chroma database

Optional ingest-only mode from existing raw text:

```bash
uv run python collect_papers_to_chroma.py --ingest-only --ingest-from-raw-text
```

## File 2: Run the Streamlit app

After the Chroma database has been created, start the app:

```bash
uv run streamlit run main.py
```

Then open the local URL shown by Streamlit in your browser, usually:

```text
http://localhost:8501
```

## Environment variables

The app reads these optional environment variables:

- `CHROMA_DIR` default: `./chroma_db`
- `COLLECTION_NAME` default: `research_papers_rag`
- `OLLAMA_MODEL` default: `llama3.2:3b`
- `TOP_K` default: `10`
- `MAX_CONTEXT_CHARS` default: `12000`

Example:

```bash
OLLAMA_MODEL=llama3.2:3b uv run streamlit run main.py
```

## Typical workflow

```bash
uv sync
uv run python collect_papers_to_chroma.py
uv run streamlit run main.py
```

## Troubleshooting

`main.py` fails because the collection does not exist:
- Run `uv run python collect_papers_to_chroma.py` first.

Ollama model errors:
- Make sure Ollama is running.
- Make sure the selected model is installed locally with `ollama pull`.

No papers or chunks found:
- Check your network connection for arXiv downloads.
- Try adjusting `--max-papers`, `--start-year`, or `--end-year`.
