# Developer Guide: Hybrid Physics RAG System

This guide covers the section-aware ingestion pipeline, the multi-layered retrieval strategy (Vector, BM25 Lexical, and Relational Graph), Reciprocal Rank Fusion (RRF) rankings, and the observability diagnostics panel implemented for the NCERT Physics Part 1 Tutor.

---

## 1. Pipeline Ingestion & Extraction Flow

Educational textbooks contain dense diagrams, formulas, and headers that standard character-limit chunkers split arbitrarily, destroying context. We implement a **Section-Aware Ingestion Pipeline**:

```
                       [NCERT Physics PDF / Database]
                                     │
                                     ▼
                       [Page-by-Page Ingestion]
                        Extracts text and page tags
                                     │
                                     ▼
                        [Heading Regex Filtering]
                     Groups text by Chapter & Section
                                     │
                                     ▼
                      [Equation Extraction & Parser]
                    Normalizes mathematical equations
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
             [Vector DB]           [BM25]           [Graph DB]
            FAISS Index /        Lexical term      NetworkX Relational
          TF-IDF Cosine Space      frequencies         Concept Map
```

---

## 2. File-by-File Blueprint

All retrieval components reside in `hybrid_rag/src/`:

### A. [ingest.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/hybrid_rag/src/ingest.py)
* `download_physics_pdf(output_path)`: Downloads target chapters from the NCERT website using custom user-agent headers. If offline, catches network errors gracefully.
* `parse_pdf(pdf_path)`: Uses `pdfplumber` to parse pages, extract section headings (e.g. `1.3 Coulomb's Law`), isolate formula lines containing variables like $F$, $E$, $V$, or $\varepsilon_0$, and append page tags.
* `MOCK_PHYSICS_DATABASE`: Acts as a zero-dependency offline fallback, storing the actual textbook pages, text, and LaTeX equations from NCERT Part 1 (Chapters 1, 2, 3, 4, 6) to keep search scripts fully functional offline.

### B. [vector_db.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/hybrid_rag/src/vector_db.py)
* `TFIDFVectorizer`: A pure-NumPy class implementing a sparse term vector space. It normalizes term vectors (L2 Norm) and calculates Cosine similarity via dot products.
* `VectorStoreManager`: Manages semantic retrieval. It initializes a FAISS Inner Product index (`faiss.IndexFlatIP`) when available, adding normalized document vectors. If FAISS is not installed, it falls back to raw NumPy matrix multiplications:
  $$\text{Scores} = X_{tfidf} \cdot q^T$$

### C. [graph_db.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/hybrid_rag/src/graph_db.py)
* `CustomGraphDB`: A zero-dependency graph representation mapping nodes and directed/undirected relationships.
* `KnowledgeGraphManager`: Coordinates walks using NetworkX. Builds a structural map of the NCERT book (Chapters, Topics, Concepts, Formulas, Pages).
* `search_graph(query, max_depth)`: Tokenizes the query and excludes generic stop-words (e.g. "explain", "page", "formula", "law") to avoid over-matching. Matches the remaining unique tokens against concept nodes (e.g., "Coulomb" -> "Coulomb's Law" node), and traverses adjacent edges via Breadth-First Search (BFS) to retrieve exact linked formulas and page numbers.

### D. [search.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/hybrid_rag/src/search.py)
* `BM25Indexer`: A pure-Python first-principles BM25 lexical retriever.
* `HybridSearchRanker`: Merges Vector, BM25, and Graph traversals. Aggregates vector and keyword ranks using Reciprocal Rank Fusion (RRF), expands the context with graph-derived formulas, and compiles a diagnostics payload.

### E. [rag_chain.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/hybrid_rag/src/rag_chain.py)
* `RAGChain`: Formulates prompts with strict grounding rules, page citation requirements, and refusal instructions for out-of-domain queries. If a Gemini API key is configured, it sends requests via a zero-dependency `urllib.request.urlopen` connection. If offline, it maps input keywords to a pre-defined textbook mock response.

### F. [server.py](file:///home/Krishna-Singh/vanco-ai-solution-architect/hybrid_rag/src/server.py)
* Serves a glassmorphism dark-mode UI containing a dual-pane workspace: a Chatbot column on the left and an Observability Diagnostics Panel on the right (inspecting Vector Cosine matches, BM25 scores, BFS Graph walk steps, and the final compiled LLM prompt).

---

## 3. Core Algorithms & Math Formulation

### A. BM25 Lexical Score
For each chunk $D$ and query terms $q_i$, the lexical score is calculated as:
$$\text{Score}_{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* $f(q_i, D)$ is the term frequency in document $D$.
* $|D|$ and $\text{avgdl}$ represent document length and average corpus length.
* $\text{IDF}(q_i) = \log\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$ ensures unique terms (like "Lenz") dominate common words.

### B. Reciprocal Rank Fusion (RRF)
RRF merges ranks rather than scores to bypass differing scales between vector space distances and keyword tallies:
$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
* $M$ represents the set of retrieval models (Vector and BM25).
* $r_m(d)$ is the rank of document $d$ within index run $m$ (1-indexed).
* $k = 60$ is a standard smoothing constant preventing low ranks from disproportionately penalizing candidates.

### C. Graph Walk Context Injection
For a query about "Gauss Law", the graph manager:
1. Matches token "gauss" to node `"Gauss's Law"` (Topic).
2. Traverses adjacent edges to locate `"Gauss Formula"` (Formula) and `"Page 33"` (Page).
3. Injects the LaTeX string `\oint \mathbf{E} \cdot d\mathbf{A} = \frac{q}{\varepsilon_0}` directly into the prompt context, bypassing standard text chunking limits.

---

## 4. Failure Modes & Limitations

1. **PDF Mathematical Symbol Garbling**: NCERT equations contain integrals ($\oint$, $\int$), square roots, and Greek symbols ($\varepsilon_0$, $\Phi$). Standard PDF parsers often extract these as empty boxes or garbled strings. We address this by compiling a clean math regex translator and a fallback database.
2. **RRF Rank Mismatch**: A chunk with excellent keyword similarity (BM25 Rank 1) but poor embedding representation (Vector Rank 20) can score lower than a document with average ranks in both runs (e.g. Vector Rank 5, BM25 Rank 5) due to RRF rank summation constraints.
3. **Multi-Hop Traversal Spans**: If BFS traversals exceed `depth = 2` without strict term constraints, graph walks can cross-reference chapters and pull in unrelated physics formulas, polluting the prompt length limit.
