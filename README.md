# Research Paper RAG

This project has two main Python files:

- `collect_papers_to_chroma.py`: collects research papers, extracts text, chunks it, and stores embeddings in a local Chroma database.
- `main.py`: starts the Streamlit app with both a chat interface and an evaluation interface.

Run them in this order:

1. Build or refresh the Chroma database with `collect_papers_to_chroma.py`
2. Start the app with `main.py`

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) recommended for dependency management
- An OpenAI API key available in `OPENAI_API_KEY`

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
OPENAI_API_KEY=your_key_here uv run streamlit run main.py
```

Then open the local URL shown by Streamlit in your browser, usually:

```text
http://localhost:8501
```

The app has two tabs:

- `Chat`: asks questions over the local Chroma collection
- `Evaluation`: reads your evaluation JSON file, shows each question with `rag_answer` and `llm_answer`, and computes accuracy for both

## Evaluation file

By default, the Streamlit evaluation tab reads:

```text
./paper_dataset/random_20_qa_eval.json
```

Each JSON item should include these fields:

- `paper_arxiv_id`
- `paper_title`
- `pdf_path`
- `question`
- `ground_truth_answer`
- `rag_answer`
- `llm_answer`
- `rag_correct`
- `llm_correct`

`rag_correct` and `llm_correct` are interpreted as boolean-like values. These work:

- `1`, `true`, `yes`
- `0`, `false`, `no`

Example item:

```json
{
  "paper_arxiv_id": "2312.03633v3",
  "paper_title": "Exploring the Reversal Curse and Other Deductive Logical Reasoning in BERT and GPT-Based Large Language Models",
  "pdf_path": "paper_dataset/pdfs/2312.03633v3.pdf",
  "question": "Which language model was reported to be immune to the reversal curse in experiments comparing BERT and GPT-style models?",
  "ground_truth_answer": "BERT was found to be immune to the reversal curse.",
  "rag_answer": "BERT",
  "llm_answer": "BERT was found to be immune to the reversal curse.",
  "rag_correct": "1",
  "llm_correct": "1"
}
```

## Environment variables

The app reads these optional environment variables:

- `CHROMA_DIR` default: `./chroma_db`
- `COLLECTION_NAME` default: `research_papers_rag`
- `TOP_K` default: `5`
- `MAX_CONTEXT_CHARS` default: `6000`
- `EVAL_JSON_PATH` default: `./paper_dataset/random_20_qa_eval.json`
- `OPENAI_API_KEY` required for answer generation

The app currently uses:

- OpenAI model: `gpt-4o-mini`

Example:

```bash
OPENAI_API_KEY=your_key_here \
EVAL_JSON_PATH=./paper_dataset/random_20_qa_eval.json \
uv run streamlit run main.py
```

## Typical workflow

```bash
uv sync
uv run python collect_papers_to_chroma.py
OPENAI_API_KEY=your_key_here uv run streamlit run main.py
```

If you already prepared evaluation results, place them in `paper_dataset/random_20_qa_eval.json` or point the app to a different file with `EVAL_JSON_PATH`.

## Troubleshooting

`main.py` fails because the collection does not exist:
- Run `uv run python collect_papers_to_chroma.py` first.

OpenAI errors:
- Make sure `OPENAI_API_KEY` is set.
- Make sure your API key has access to the configured model.

Evaluation tab shows no rows:
- Check that the JSON file exists.
- Check that the file is a JSON array.
- Check that `EVAL_JSON_PATH` points to the correct file.

No papers or chunks found:
- Check your network connection for arXiv downloads.
- Try adjusting `--max-papers`, `--start-year`, or `--end-year`.
