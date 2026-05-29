import math
import re
import json

class BM25Indexer:
    """
    A pure-Python first-principles implementation of BM25.
    Requires no external packages, ensuring 100% reproducibility.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_lens = []
        self.doc_term_freqs = [] # list of dicts: term -> count
        self.df = {} # term -> document frequency
        
    def _tokenize(self, text):
        return re.findall(r'[a-z0-9]+', text.lower())
        
    def fit(self, chunks):
        self.chunks = chunks
        self.corpus_size = len(chunks)
        
        total_len = 0
        self.doc_lens = []
        self.doc_term_freqs = []
        self.df = {}
        
        for chunk in chunks:
            tokens = self._tokenize(chunk['text'])
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            
            # Count terms in this doc
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)
            
            # Document frequency
            for token in tf.keys():
                self.df[token] = self.df.get(token, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
    def _idf(self, term):
        df_t = self.df.get(term, 0)
        # Standard BM25 IDF formula with smoothing
        return math.log((self.corpus_size - df_t + 0.5) / (df_t + 0.5) + 1.0)
        
    def search(self, query, k=3):
        """
        Calculates BM25 score for all documents against the query.
        Returns list of tuples: (chunk_dict, score) sorted by score.
        """
        query_terms = self._tokenize(query)
        scores = []
        
        for doc_idx, tf in enumerate(self.doc_term_freqs):
            score = 0.0
            doc_len = self.doc_lens[doc_idx]
            
            for term in query_terms:
                if term in tf:
                    tf_t = tf[term]
                    idf_t = self._idf(term)
                    
                    # BM25 term weighting formula
                    numerator = tf_t * (self.k1 + 1)
                    denominator = tf_t + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf_t * (numerator / denominator)
                    
            scores.append((self.chunks[doc_idx], score))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class HybridSearchRanker:
    """
    Orchestrates the Hybrid Retrieval Pipeline:
      1. Performs Semantic Vector Search (FAISS/TF-IDF)
      2. Performs Exact Keyword Search (BM25)
      3. Performs Knowledge Graph Expansion (NetworkX)
      4. Merges Vector & BM25 ranks using Reciprocal Rank Fusion (RRF)
      5. Integrates Graph formulas and cross-chapter linkages
    """
    def __init__(self, chunks, vector_manager, graph_manager):
        self.chunks = chunks
        self.vector_manager = vector_manager
        self.graph_manager = graph_manager
        
        # Initialize BM25 Indexer
        self.bm25 = BM25Indexer()
        self.bm25.fit(chunks)
        
    def search(self, query, k=3, rrf_constant=60):
        print(f"\n--- Hybrid Search Engine Executing Query: '{query}' ---")
        
        # 1. BM25 Lexical Retrieve
        bm25_results = self.bm25.search(query, k=10)
        
        # 2. Vector Semantic Retrieve
        vector_results = self.vector_manager.search(query, k=10)
        
        # 3. Knowledge Graph Expansion
        graph_results = self.graph_manager.search_graph(query)
        
        # 4. Reciprocal Rank Fusion (RRF)
        # Create map of chunk_text -> chunk dict to ensure unique identity
        # and accumulate RRF scores
        rrf_scores = {} # text -> {chunk: chunk_dict, score: rrf_score}
        
        # Helper to compute unique identifier
        def get_chunk_key(chunk):
            return f"{chunk['chapter']}_{chunk['section']}_p{chunk['page']}"
            
        # Process Vector ranks
        for rank, (chunk, score) in enumerate(vector_results):
            key = get_chunk_key(chunk)
            score_contrib = 1.0 / (rrf_constant + (rank + 1))
            if key not in rrf_scores:
                rrf_scores[key] = {"chunk": chunk, "score": 0.0, "vector_rank": rank + 1, "bm25_rank": None}
            rrf_scores[key]["score"] += score_contrib
            rrf_scores[key]["vector_rank"] = rank + 1
            
        # Process BM25 ranks
        for rank, (chunk, score) in enumerate(bm25_results):
            key = get_chunk_key(chunk)
            score_contrib = 1.0 / (rrf_constant + (rank + 1))
            if key not in rrf_scores:
                rrf_scores[key] = {"chunk": chunk, "score": 0.0, "vector_rank": None, "bm25_rank": rank + 1}
            rrf_scores[key]["score"] += score_contrib
            rrf_scores[key]["bm25_rank"] = rank + 1
            
        # Sort merged documents by RRF score descending
        merged_results = list(rrf_scores.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Take Top K
        top_k_merged = merged_results[:k]
        
        # 5. Graph-RAG Context Injection
        # We inject Graph-derived formulas, adjacent chapters, and concepts into the final context packet
        print(f"Graph Search matched {len(graph_results['concepts'])} concepts and {len(graph_results['formulas'])} formulas.")
        
        # Format diagnostics for the dashboard
        diagnostics = {
            "query": query,
            "vector_search": [
                {"chapter": c["chapter"], "section": c["section"], "page": c["page"], "score": s}
                for c, s in vector_results[:4]
            ],
            "bm25_search": [
                {"chapter": c["chapter"], "section": c["section"], "page": c["page"], "score": s}
                for c, s in bm25_results[:4]
            ],
            "graph_traversal": {
                "concepts": graph_results["concepts"],
                "formulas": graph_results["formulas"],
                "pages": graph_results["pages"],
                "steps": graph_results["steps"]
            },
            "rrf_merged": [
                {
                    "chapter": d["chunk"]["chapter"],
                    "section": d["chunk"]["section"],
                    "page": d["chunk"]["page"],
                    "rrf_score": d["score"],
                    "vector_rank": d["vector_rank"],
                    "bm25_rank": d["bm25_rank"]
                }
                for d in top_k_merged
            ]
        }
        
        # Final consolidated retrieval results
        retrieved_chunks = [d["chunk"] for d in top_k_merged]
        
        return retrieved_chunks, graph_results["formulas"], diagnostics

if __name__ == '__main__':
    from ingest import get_chunks
    from vector_db import VectorStoreManager
    from graph_db import build_physics_knowledge_graph
    
    chunks = get_chunks()
    
    vm = VectorStoreManager()
    vm.build_index(chunks)
    
    kg = build_physics_knowledge_graph()
    
    ranker = HybridSearchRanker(chunks, vm, kg)
    
    query = "What is Coulomb's Law formula and its page?"
    retrieved_chunks, formulas, diagnostics = ranker.search(query, k=2)
    
    print("\nTop Merged RRF Chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. [Page {chunk['page']}] {chunk['chapter']} - {chunk['section']}")
        
    print("\nInjected Graph Formulas:")
    for f in formulas:
        print(f"  - {f}")
