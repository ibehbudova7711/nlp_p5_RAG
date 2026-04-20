import argparse
import hashlib
import json
import re
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chromadb
import requests
import tabula
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from urllib3.util.retry import Retry


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}.pdf"

DEFAULT_QUERIES = [
    'cat:cs.CL AND (all:transformer OR all:"large language model" OR all:bert OR all:gpt)',
    'cat:cs.CL AND (all:"question answering" OR all:summarization OR all:"machine translation")',
    'cat:cs.CL AND (all:"natural language processing" OR all:"language model")',
    'cat:cs.LG AND (all:"deep learning" OR all:"representation learning" OR all:"self-supervised learning")',
    'cat:cs.LG AND (all:"diffusion model" OR all:"contrastive learning" OR all:"reinforcement learning")',
    'cat:cs.AI AND (all:"foundation model" OR all:"reasoning" OR all:"retrieval augmented generation")',
    'cat:stat.ML AND (all:"machine learning" OR all:"large-scale training" OR all:"optimization")',
]

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    summary: str
    published: str
    updated: str
    pdf_url: str
    categories: List[str]
    authors: List[str]
    primary_category: str


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "paper-rag-collector/1.0 (research dataset builder)"}
    )
    return session


class ArxivCollector:
    def __init__(self, session: requests.Session, delay_seconds: float = 3.0) -> None:
        self.session = session
        self.delay_seconds = delay_seconds

    def _request_with_retry(self, params: dict, max_attempts: int = 5) -> requests.Response:
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.get(ARXIV_API_URL, params=params, timeout=60)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_err = e
                sleep_s = min(2 ** attempt, 20)
                print(f"[warn] arXiv request failed (attempt {attempt}/{max_attempts}): {e}")
                time.sleep(sleep_s)
        raise last_err

    def _parse_feed(self, xml_text: str) -> List[Paper]:
        root = ET.fromstring(xml_text)
        papers: List[Paper] = []

        for entry in root.findall("atom:entry", ATOM_NS):
            raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

            title = entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS).strip()
            published = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
            updated = entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip()

            pdf_url = ""
            for link in entry.findall("atom:link", ATOM_NS):
                link_title = link.attrib.get("title", "")
                link_type = link.attrib.get("type", "")
                href = link.attrib.get("href", "")
                if link_title == "pdf" or link_type == "application/pdf":
                    pdf_url = href
                    break

            if not pdf_url:
                pdf_url = ARXIV_PDF_URL.format(paper_id=arxiv_id)

            categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ATOM_NS)]

            authors = []
            for author in entry.findall("atom:author", ATOM_NS):
                name = author.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
                if name:
                    authors.append(name)

            primary_category_elem = entry.find("arxiv:primary_category", ATOM_NS)
            primary_category = ""
            if primary_category_elem is not None:
                primary_category = primary_category_elem.attrib.get("term", "")

            if arxiv_id and title:
                papers.append(
                    Paper(
                        arxiv_id=arxiv_id,
                        title=title,
                        summary=summary,
                        published=published,
                        updated=updated,
                        pdf_url=pdf_url,
                        categories=categories,
                        authors=authors,
                        primary_category=primary_category,
                    )
                )

        return papers

    def _search_single_query(
        self,
        query: str,
        start_year: int,
        end_year: int,
        max_results: int,
        batch_size: int = 25,
        max_offset: int = 500,
    ) -> List[Paper]:
        papers: List[Paper] = []
        start = 0
        seen_ids = set()

        while len(papers) < max_results and start < max_offset:
            params = {
                "search_query": query,
                "start": start,
                "max_results": batch_size,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            response = self._request_with_retry(params)
            batch = self._parse_feed(response.text)

            if not batch:
                break

            new_added = 0
            for paper in batch:
                if paper.arxiv_id in seen_ids:
                    continue

                try:
                    year = int(paper.published[:4])
                except Exception:
                    continue

                if start_year <= year <= end_year:
                    seen_ids.add(paper.arxiv_id)
                    papers.append(paper)
                    new_added += 1
                    if len(papers) >= max_results:
                        break

            if new_added == 0 and len(batch) < batch_size:
                break

            start += batch_size
            time.sleep(self.delay_seconds)

        return papers

    def search(
        self,
        queries: List[str],
        start_year: int,
        end_year: int,
        max_results: int,
        per_query_limit: int = 40,
    ) -> List[Paper]:
        all_papers: List[Paper] = []
        seen_ids = set()

        for q in queries:
            print(f"[info] searching arXiv with: {q}")
            try:
                batch = self._search_single_query(
                    query=q,
                    start_year=start_year,
                    end_year=end_year,
                    max_results=per_query_limit,
                    batch_size=25,
                    max_offset=500,
                )
            except requests.RequestException as e:
                print(f"[warn] skipping query due to repeated failure: {q} | {e}")
                continue

            for paper in batch:
                if paper.arxiv_id in seen_ids:
                    continue
                seen_ids.add(paper.arxiv_id)
                all_papers.append(paper)

        priority_terms = [
            "transformer",
            "bert",
            "gpt",
            "large language",
            "diffusion",
            "retrieval",
            "self-supervised",
            "contrastive",
            "foundation model",
            "instruction",
            "reasoning",
            "language model",
        ]

        def score(p: Paper) -> tuple:
            title_lower = p.title.lower()
            summary_lower = p.summary.lower()
            keyword_hits = sum(term in title_lower for term in priority_terms) * 3
            keyword_hits += sum(term in summary_lower for term in priority_terms)
            year = int(p.published[:4]) if p.published[:4].isdigit() else 0
            author_bonus = min(len(p.authors), 6)
            return (keyword_hits, year, author_bonus)

        all_papers.sort(key=score, reverse=True)
        return all_papers[:max_results]


class TextPreprocessor:
    @staticmethod
    def remove_invalid_unicode(text: str) -> str:
        if not text:
            return ""
        return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    @staticmethod
    def clean_text(text: str) -> str:
        text = TextPreprocessor.remove_invalid_unicode(text)
        text = text.replace("\x00", " ")
        text = re.sub(r"-\n(?=\w)", "", text)
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if re.fullmatch(r"\d+", stripped):
                continue
            lines.append(stripped)

        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = TextPreprocessor.remove_invalid_unicode(text)
        return text.strip()

    @staticmethod
    def clean_cell(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = TextPreprocessor.remove_invalid_unicode(text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))


class PDFStructuredExtractor:
    def __init__(
        self,
        session: requests.Session,
        pdf_dir: Path,
        use_tabula: bool = True,
        tabula_pages: str = "all",
        tabula_lattice: bool = False,
        tabula_stream: bool = True,
    ) -> None:
        self.session = session
        self.pdf_dir = pdf_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.use_tabula = use_tabula
        self.tabula_pages = tabula_pages
        self.tabula_lattice = tabula_lattice
        self.tabula_stream = tabula_stream

    def _download_pdf(self, paper: Paper) -> Path:
        pdf_path = self.pdf_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        if pdf_path.exists():
            return pdf_path

        response = self.session.get(paper.pdf_url, timeout=120)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)
        return pdf_path

    def _extract_page_texts(self, pdf_path: Path) -> List[str]:
        reader = PdfReader(str(pdf_path))
        page_texts: List[str] = []

        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:
                page_texts.append("")

        return page_texts

    def _extract_tables_by_page(self, pdf_path: Path) -> Dict[int, List[str]]:
        if not self.use_tabula:
            return {}

        tables_by_page: Dict[int, List[str]] = {}

        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
        except Exception:
            return {}

        for page_num in range(1, num_pages + 1):
            try:
                dfs = tabula.read_pdf(
                    str(pdf_path),
                    pages=page_num,
                    multiple_tables=True,
                    lattice=self.tabula_lattice,
                    stream=self.tabula_stream,
                    guess=True,
                    silent=True,
                )
            except Exception as e:
                print(f"[warn] Tabula failed on page {page_num} of {pdf_path.name}: {e}")
                continue

            if not dfs:
                continue

            page_tables: List[str] = []
            for table_idx, df in enumerate(dfs, start=1):
                table_text = self._dataframe_to_table_block(df, page_num=page_num, table_idx=table_idx)
                if table_text:
                    page_tables.append(table_text)

            if page_tables:
                tables_by_page[page_num] = page_tables

        return tables_by_page

    def _dataframe_to_table_block(self, df, page_num: int, table_idx: int) -> str:
        try:
            if df is None or df.empty:
                return ""
        except Exception:
            return ""

        df = df.fillna("")

        headers = [TextPreprocessor.clean_cell(col) for col in df.columns.tolist()]
        headers = [h if h else f"column_{i+1}" for i, h in enumerate(headers)]

        rows: List[str] = []
        for row_idx, row in enumerate(df.values.tolist(), start=1):
            cleaned_cells = [TextPreprocessor.clean_cell(cell) for cell in row]
            if not any(cleaned_cells):
                continue

            pairs = [f"{headers[i]}: {cleaned_cells[i]}" for i in range(min(len(headers), len(cleaned_cells)))]
            row_text = " | ".join(pairs).strip()
            if row_text:
                rows.append(f"Row {row_idx}: {row_text}")

        if not rows:
            return ""

        header_line = " | ".join(headers)

        return (
            f"\n[TABLE page={page_num} index={table_idx}]\n"
            f"Columns: {header_line}\n"
            + "\n".join(rows)
            + f"\n[/TABLE]\n"
        )

    def extract_text_with_tables(self, paper: Paper) -> Tuple[str, Dict[str, int], Path]:
        pdf_path = self._download_pdf(paper)
        page_texts = self._extract_page_texts(pdf_path)
        tables_by_page = self._extract_tables_by_page(pdf_path)

        merged_pages: List[str] = []
        total_tables = 0

        for page_idx, page_text in enumerate(page_texts, start=1):
            page_text = page_text or ""
            table_blocks = tables_by_page.get(page_idx, [])
            total_tables += len(table_blocks)

            merged = f"[PAGE {page_idx}]\n{page_text.strip()}\n"
            if table_blocks:
                merged += "\n" + "\n".join(table_blocks)
            merged_pages.append(merged.strip())

        full_text = "\n\n".join(merged_pages)

        stats = {
            "page_count": len(page_texts),
            "table_count": total_tables,
        }
        return full_text, stats, pdf_path


class RecursiveTokenChunker:
    TABLE_BLOCK_PATTERN = re.compile(r"(\[TABLE.*?\].*?\[/TABLE\])", re.DOTALL)

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 350,
        chunk_overlap: int = 60,
        min_chunk_tokens: int = 200,
        max_chunk_tokens: int = 500,
    ) -> None:
        self.tokenizer_model = SentenceTransformer(model_name)
        self.tokenizer = self.tokenizer_model.tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens

    def estimate_tokens(self, text: str) -> int:
        text = text.strip()
        if not text:
            return 0
        words = len(text.split())
        return max(1, int(words * 1.3))

    def safe_token_count(self, text: str) -> int:
        text = text.strip()
        if not text:
            return 0
        if len(text) > 2000:
            return self.estimate_tokens(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        blocks = self._split_preserving_tables(text)
        coarse_chunks: List[str] = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if self._is_table_block(block):
                coarse_chunks.append(block)
            else:
                coarse_chunks.extend(self._recursive_split_by_structure(block))

        merged_chunks = self._merge_small_chunks_table_aware(coarse_chunks)

        final_chunks: List[str] = []
        for chunk in merged_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            est = self.safe_token_count(chunk)

            if est < self.min_chunk_tokens:
                if self._is_table_block(chunk):
                    final_chunks.append(chunk)
                continue

            if est > self.max_chunk_tokens:
                if self._is_table_block(chunk):
                    final_chunks.extend(self._split_large_table_block(chunk))
                else:
                    final_chunks.extend(self._split_large_chunk(chunk))
            else:
                final_chunks.append(chunk)

        return self._add_overlap(final_chunks)

    def _split_preserving_tables(self, text: str) -> List[str]:
        parts = self.TABLE_BLOCK_PATTERN.split(text)
        return [p for p in parts if p and p.strip()]

    def _is_table_block(self, text: str) -> bool:
        return text.strip().startswith("[TABLE") and text.strip().endswith("[/TABLE]")

    def _recursive_split_by_structure(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if self.estimate_tokens(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._split_large_chunk(text)

        sep = separators[0]
        parts = text.split(sep)

        if len(parts) == 1:
            return self._recursive_split(text, separators[1:])

        results: List[str] = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            candidate = part if not current else current + sep + part

            if self.estimate_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    results.extend(self._recursive_split(current, separators[1:]))
                current = part

        if current.strip():
            results.extend(self._recursive_split(current, separators[1:]))

        return results

    def _merge_small_chunks_table_aware(self, pieces: List[str]) -> List[str]:
        if not pieces:
            return []

        merged: List[str] = []
        current = pieces[0].strip()

        for nxt in pieces[1:]:
            nxt = nxt.strip()
            if not nxt:
                continue

            if self._is_table_block(current) or self._is_table_block(nxt):
                if self.estimate_tokens(current) < self.min_chunk_tokens:
                    candidate = current + "\n\n" + nxt
                    if self.estimate_tokens(candidate) <= self.max_chunk_tokens:
                        current = candidate
                    else:
                        merged.append(current)
                        current = nxt
                else:
                    merged.append(current)
                    current = nxt
                continue

            if self.estimate_tokens(current) < self.min_chunk_tokens:
                candidate = current + " " + nxt
                if self.estimate_tokens(candidate) <= self.max_chunk_tokens:
                    current = candidate
                else:
                    merged.append(current)
                    current = nxt
            else:
                merged.append(current)
                current = nxt

        if current.strip():
            merged.append(current)

        return merged

    def _split_large_chunk(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        approx_words_per_chunk = max(50, int(self.chunk_size / 1.3))
        approx_words_overlap = max(10, int(self.chunk_overlap / 1.3))

        chunks: List[str] = []
        start = 0

        while start < len(words):
            end = min(start + approx_words_per_chunk, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)

            if end == len(words):
                break

            start = max(0, end - approx_words_overlap)

        return chunks

    def _split_large_table_block(self, table_block: str) -> List[str]:
        lines = [line.strip() for line in table_block.splitlines() if line.strip()]
        if len(lines) <= 4:
            return [table_block]

        open_line = lines[0]
        close_line = lines[-1]
        middle = lines[1:-1]

        columns_line = middle[0] if middle and middle[0].startswith("Columns:") else "Columns:"
        row_lines = middle[1:] if middle else []

        chunks: List[str] = []
        current_rows: List[str] = []
        current_text = f"{open_line}\n{columns_line}\n{close_line}"

        for row in row_lines:
            candidate_rows = current_rows + [row]
            candidate = f"{open_line}\n{columns_line}\n" + "\n".join(candidate_rows) + f"\n{close_line}"

            if self.estimate_tokens(candidate) <= self.max_chunk_tokens:
                current_rows.append(row)
                current_text = candidate
            else:
                if current_rows:
                    chunks.append(current_text)
                current_rows = [row]
                current_text = f"{open_line}\n{columns_line}\n{row}\n{close_line}"

        if current_rows:
            chunks.append(current_text)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped: List[str] = [chunks[0]]
        approx_words_overlap = max(10, int(self.chunk_overlap / 1.3))

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            if self._is_table_block(curr):
                overlapped.append(curr)
                continue

            prev_words = prev.split()
            curr_words = curr.split()

            overlap_words = prev_words[-approx_words_overlap:] if len(prev_words) > approx_words_overlap else prev_words
            combined = " ".join(overlap_words + curr_words).strip()

            if self.estimate_tokens(combined) > self.max_chunk_tokens:
                combined_words = combined.split()
                max_words = max(50, int(self.max_chunk_tokens / 1.3))
                combined = " ".join(combined_words[-max_words:])

            overlapped.append(combined)

        return overlapped


class ChromaIndexer:
    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        recreate_collection: bool = True,
    ) -> None:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

        existing = [c.name for c in self.client.list_collections()]

        if collection_name in existing and recreate_collection:
            self.client.delete_collection(collection_name)

        current_names = [c.name for c in self.client.list_collections()]
        if collection_name in current_names:
            self.collection = self.client.get_collection(name=collection_name)
        else:
            self.collection = self.client.create_collection(name=collection_name)

        try:
            self.max_batch_size = int(self.client.get_max_batch_size())
        except Exception:
            self.max_batch_size = 5000

        print(f"[info] Chroma max batch size: {self.max_batch_size}")

    def add_documents(self, docs: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        total = len(docs)
        if total == 0:
            return

        for start in range(0, total, self.max_batch_size):
            end = min(start + self.max_batch_size, total)
            batch_docs = docs[start:end]
            batch_metas = metadatas[start:end]
            batch_ids = ids[start:end]

            print(f"[info] embedding batch {start}:{end} / {total}")
            batch_embeddings = self.embedding_model.encode(
                batch_docs,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).tolist()

            print(f"[info] inserting batch {start}:{end} / {total}")
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embeddings,
            )


def stable_chunk_id(paper_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{paper_id}:{chunk_index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{paper_id.replace('/', '_')}_chunk_{chunk_index}_{digest}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metadata(output_path: Path, papers: List[Paper], preprocessor: TextPreprocessor) -> None:
    serializable = []
    for p in papers:
        item = asdict(p)
        for k, v in item.items():
            if isinstance(v, str):
                item[k] = preprocessor.remove_invalid_unicode(v)
            elif isinstance(v, list):
                item[k] = [
                    preprocessor.remove_invalid_unicode(x) if isinstance(x, str) else x
                    for x in v
                ]
        serializable.append(item)

    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_chunks(
    papers: List[Paper],
    extractor: PDFStructuredExtractor,
    preprocessor: TextPreprocessor,
    chunker: RecursiveTokenChunker,
    raw_text_dir: Path,
) -> Iterable[Dict]:
    for idx, paper in enumerate(papers, start=1):
        print(f"[info] processing paper {idx}/{len(papers)}: {paper.arxiv_id} | {paper.title}")

        try:
            raw_text_with_tables, extract_stats, pdf_path = extractor.extract_text_with_tables(paper)
        except Exception as e:
            print(f"[warn] failed to extract {paper.arxiv_id}: {e}")
            continue

        try:
            cleaned_text = preprocessor.clean_text(raw_text_with_tables)
        except Exception as e:
            print(f"[warn] cleaning failed for {paper.arxiv_id}: {e}")
            continue

        words = preprocessor.word_count(cleaned_text)
        if words < 500:
            print(f"[warn] skipping very short extraction for {paper.arxiv_id}")
            continue

        raw_path = raw_text_dir / f"{paper.arxiv_id.replace('/', '_')}.txt"

        try:
            safe_text = preprocessor.remove_invalid_unicode(cleaned_text)
            raw_path.write_text(safe_text, encoding="utf-8")
        except UnicodeEncodeError as e:
            print(f"[warn] skipping {paper.arxiv_id} due to unicode write error: {e}")
            continue
        except Exception as e:
            print(f"[warn] failed saving cleaned text for {paper.arxiv_id}: {e}")
            continue

        try:
            chunks = chunker.split_text(cleaned_text)
        except Exception as e:
            print(f"[warn] chunking failed for {paper.arxiv_id}: {e}")
            continue

        if not chunks:
            print(f"[warn] no chunks produced for {paper.arxiv_id}")
            continue

        for chunk_index, chunk_text in enumerate(chunks):
            try:
                chunk_text = preprocessor.remove_invalid_unicode(chunk_text)
                token_count = chunker.safe_token_count(chunk_text)
                has_table = "[TABLE" in chunk_text and "[/TABLE]" in chunk_text
            except Exception as e:
                print(f"[warn] skipping chunk {chunk_index} of {paper.arxiv_id}: {e}")
                continue

            yield {
                "id": stable_chunk_id(paper.arxiv_id, chunk_index, chunk_text),
                "document": chunk_text,
                "metadata": {
                    "arxiv_id": paper.arxiv_id,
                    "title": preprocessor.remove_invalid_unicode(paper.title),
                    "summary": preprocessor.remove_invalid_unicode(paper.summary),
                    "published": paper.published,
                    "updated": paper.updated,
                    "pdf_url": paper.pdf_url,
                    "pdf_local_path": str(pdf_path),
                    "categories": preprocessor.remove_invalid_unicode(", ".join(paper.categories)),
                    "primary_category": preprocessor.remove_invalid_unicode(paper.primary_category),
                    "authors": preprocessor.remove_invalid_unicode(", ".join(paper.authors)),
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                    "word_count": preprocessor.word_count(chunk_text),
                    "source_text_path": str(raw_path),
                    "has_table": has_table,
                    "paper_table_count": int(extract_stats.get("table_count", 0)),
                    "paper_page_count": int(extract_stats.get("page_count", 0)),
                },
            }


def load_metadata_map(metadata_path: Path, preprocessor: TextPreprocessor) -> Dict[str, Dict]:
    if not metadata_path.exists():
        return {}

    try:
        items = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] could not read metadata file: {e}")
        return {}

    metadata_map = {}
    for item in items:
        arxiv_id = str(item.get("arxiv_id", "")).strip()
        if not arxiv_id:
            continue

        safe_item = {}
        for k, v in item.items():
            if isinstance(v, str):
                safe_item[k] = preprocessor.remove_invalid_unicode(v)
            elif isinstance(v, list):
                safe_item[k] = [
                    preprocessor.remove_invalid_unicode(x) if isinstance(x, str) else x
                    for x in v
                ]
            else:
                safe_item[k] = v

        metadata_map[arxiv_id] = safe_item

    return metadata_map


def load_chunks_from_raw_text(
    raw_text_dir: Path,
    metadata_path: Path,
    preprocessor: TextPreprocessor,
    chunker: RecursiveTokenChunker,
) -> Iterable[Dict]:
    metadata_map = load_metadata_map(metadata_path, preprocessor)
    txt_files = sorted(raw_text_dir.glob("*.txt"))

    if not txt_files:
        raise RuntimeError(f"No .txt files found in {raw_text_dir}")

    for txt_path in txt_files:
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            text = preprocessor.clean_text(text)
        except Exception as e:
            print(f"[warn] failed reading {txt_path.name}: {e}")
            continue

        if not text:
            print(f"[warn] empty text in {txt_path.name}")
            continue

        arxiv_id = txt_path.stem.replace("_", "/")
        meta = metadata_map.get(arxiv_id, {})

        title = meta.get("title", txt_path.stem)
        summary = meta.get("summary", "")
        published = meta.get("published", "")
        updated = meta.get("updated", "")
        pdf_url = meta.get("pdf_url", "")
        categories = meta.get("categories", [])
        categories_str = ", ".join(categories) if isinstance(categories, list) else str(categories)
        primary_category = meta.get("primary_category", "")
        authors = meta.get("authors", [])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)

        try:
            chunks = chunker.split_text(text)
        except Exception as e:
            print(f"[warn] chunking failed for {txt_path.name}: {e}")
            continue

        if not chunks:
            print(f"[warn] no chunks produced for {txt_path.name}")
            continue

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_text = preprocessor.remove_invalid_unicode(chunk_text)
            if not chunk_text.strip():
                continue

            yield {
                "id": stable_chunk_id(arxiv_id, chunk_index, chunk_text),
                "document": chunk_text,
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": preprocessor.remove_invalid_unicode(title),
                    "summary": preprocessor.remove_invalid_unicode(summary),
                    "published": published,
                    "updated": updated,
                    "pdf_url": pdf_url,
                    "categories": preprocessor.remove_invalid_unicode(categories_str),
                    "primary_category": preprocessor.remove_invalid_unicode(primary_category),
                    "authors": preprocessor.remove_invalid_unicode(authors_str),
                    "chunk_index": chunk_index,
                    "token_count": chunker.safe_token_count(chunk_text),
                    "word_count": preprocessor.word_count(chunk_text),
                    "source_text_path": str(txt_path),
                    "has_table": "[TABLE" in chunk_text and "[/TABLE]" in chunk_text,
                },
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect arXiv papers and index chunks into local Chroma DB with Tabula table injection.")
    parser.add_argument("--max-papers", type=int, default=100, help="Target number of papers to collect")
    parser.add_argument("--start-year", type=int, default=2020, help="Start year filter")
    parser.add_argument("--end-year", type=int, default=2026, help="End year filter")
    parser.add_argument(
        "--queries-json",
        default="",
        help="Optional JSON list of arXiv queries. If empty, built-in NLP/DL/ML queries are used.",
    )
    parser.add_argument(
        "--per-query-limit",
        type=int,
        default=40,
        help="How many papers to fetch per query before deduplication",
    )
    parser.add_argument("--output-dir", default="./paper_dataset", help="Directory to save dataset artifacts")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="Directory for local persistent Chroma DB")
    parser.add_argument("--collection-name", default="research_papers_rag", help="Chroma collection name")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=350, help="Target chunk size in estimated tokens")
    parser.add_argument("--chunk-overlap", type=int, default=60, help="Chunk overlap in estimated tokens")
    parser.add_argument("--min-chunk-tokens", type=int, default=200, help="Minimum allowed chunk size")
    parser.add_argument("--max-chunk-tokens", type=int, default=500, help="Maximum allowed chunk size")
    parser.add_argument("--disable-tabula", action="store_true", help="Disable Tabula table extraction")
    parser.add_argument("--tabula-lattice", action="store_true", help="Use lattice mode for Tabula")
    parser.add_argument("--tabula-stream", action="store_true", help="Use stream mode for Tabula")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Skip paper search/download/chunking and only ingest existing data into Chroma",
    )
    parser.add_argument(
        "--ingest-from-raw-text",
        action="store_true",
        help="In ingest-only mode, read .txt files from raw_text folder and chunk them again",
    )
    parser.add_argument(
        "--append-to-existing",
        action="store_true",
        help="Append to existing Chroma collection instead of recreating it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queries = DEFAULT_QUERIES
    if args.queries_json:
        queries = json.loads(args.queries_json)

    output_dir = Path(args.output_dir)
    raw_text_dir = output_dir / "raw_text"
    pdf_dir = output_dir / "pdfs"
    metadata_path = output_dir / "paper_metadata.json"
    chroma_dir = Path(args.chroma_dir)

    ensure_dir(output_dir)
    ensure_dir(raw_text_dir)
    ensure_dir(pdf_dir)
    ensure_dir(chroma_dir)

    preprocessor = TextPreprocessor()
    chunker = RecursiveTokenChunker(
        model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_tokens=args.min_chunk_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
    )

    indexer = ChromaIndexer(
        chroma_dir=chroma_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        recreate_collection=not args.append_to_existing,
    )

    docs: List[str] = []
    metas: List[Dict] = []
    ids: List[str] = []
    total_words = 0
    paper_ids_with_chunks = set()

    if args.ingest_only and args.ingest_from_raw_text:
        print(f"[info] ingesting from raw text folder: {raw_text_dir}")

        for item in load_chunks_from_raw_text(
            raw_text_dir=raw_text_dir,
            metadata_path=metadata_path,
            preprocessor=preprocessor,
            chunker=chunker,
        ):
            docs.append(item["document"])
            metas.append(item["metadata"])
            ids.append(item["id"])
            total_words += int(item["metadata"].get("word_count", 0))

            arxiv_id = item["metadata"].get("arxiv_id")
            if arxiv_id:
                paper_ids_with_chunks.add(arxiv_id)

    else:
        session = build_session()
        collector = ArxivCollector(session=session, delay_seconds=3.0)
        extractor = PDFStructuredExtractor(
            session=session,
            pdf_dir=pdf_dir,
            use_tabula=not args.disable_tabula,
            tabula_lattice=args.tabula_lattice,
            tabula_stream=args.tabula_stream or not args.tabula_lattice,
        )

        papers = collector.search(
            queries=queries,
            start_year=args.start_year,
            end_year=args.end_year,
            max_results=args.max_papers,
            per_query_limit=args.per_query_limit,
        )

        if not papers:
            raise RuntimeError("No papers were collected from arXiv.")

        save_metadata(metadata_path, papers, preprocessor)

        for item in iter_chunks(
            papers=papers,
            extractor=extractor,
            preprocessor=preprocessor,
            chunker=chunker,
            raw_text_dir=raw_text_dir,
        ):
            docs.append(item["document"])
            metas.append(item["metadata"])
            ids.append(item["id"])
            total_words += item["metadata"]["word_count"]
            paper_ids_with_chunks.add(item["metadata"]["arxiv_id"])

    if not docs:
        raise RuntimeError("No chunks found to add to Chroma.")

    print(f"[info] adding {len(docs)} chunks to Chroma")
    indexer.add_documents(docs=docs, metadatas=metas, ids=ids)

    summary = {
        "run_id": str(uuid.uuid4()),
        "mode": "ingest_from_raw_text" if (args.ingest_only and args.ingest_from_raw_text) else "full_pipeline",
        "total_chunks": len(docs),
        "total_words": total_words,
        "papers_with_chunks": len(paper_ids_with_chunks),
        "output_dir": str(output_dir.resolve()),
        "chroma_dir": str(chroma_dir.resolve()),
        "collection_name": args.collection_name,
        "embedding_model": args.embedding_model,
        "append_to_existing": args.append_to_existing,
    }

    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[done] completed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()