import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import pipeline modules
from ingest import get_chunks
from vector_db import VectorStoreManager
from graph_db import build_physics_knowledge_graph
from search import HybridSearchRanker
from rag_chain import RAGChain

ranker = None
chain = None

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>VANCO Physics RAG Tutor & Observability Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- MathJax for rendering physics formulas -->
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-primary: #6366f1;
            --accent-secondary: #a855f7;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }
        header {
            padding: 1.5rem 2rem;
            background: rgba(17, 24, 39, 0.4);
            border-bottom: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }
        header h1 {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        header .badge {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #818cf8;
            padding: 0.4rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .main-layout {
            display: flex;
            flex: 1;
            height: calc(100vh - 73px);
            overflow: hidden;
            width: 100vw;
        }
        .chat-section {
            flex: 4;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-color);
            background: rgba(15, 23, 42, 0.3);
            height: 100%;
        }
        .messages-container {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        .message-bubble {
            max-width: 80%;
            padding: 1rem 1.25rem;
            border-radius: 20px;
            font-size: 0.95rem;
            line-height: 1.5;
            position: relative;
        }
        .message-bubble.user {
            background: var(--accent-primary);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);
        }
        .message-bubble.bot {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        .input-bar {
            padding: 1.5rem 2rem;
            background: rgba(17, 24, 39, 0.5);
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .input-bar input {
            flex: 1;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            padding: 0.85rem 1.25rem;
            border-radius: 14px;
            color: white;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s;
        }
        .input-bar input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25);
        }
        .send-btn {
            background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary));
            color: white;
            border: none;
            padding: 0.85rem 1.75rem;
            border-radius: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .send-btn:hover {
            opacity: 0.9;
        }
        .dashboard-section {
            flex: 3;
            display: flex;
            flex-direction: column;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            height: 100%;
        }
        .tab-bar {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            background: rgba(17, 24, 39, 0.3);
        }
        .tab-btn {
            flex: 1;
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 1rem;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            border-bottom: 2px solid transparent;
        }
        .tab-btn.active {
            color: #818cf8;
            border-bottom-color: var(--accent-primary);
            background: rgba(99, 102, 241, 0.05);
        }
        .tab-content {
            flex: 1;
            padding: 1.5rem;
            overflow-y: auto;
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .diag-item {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.85rem;
            margin-bottom: 0.85rem;
            font-size: 0.85rem;
        }
        .diag-item h4 {
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
            color: #818cf8;
            display: flex;
            justify-content: space-between;
        }
        .diag-item .score {
            font-family: monospace;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.1rem 0.3rem;
            border-radius: 4px;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .diag-item p {
            color: var(--text-muted);
            line-height: 1.4;
        }
        .graph-walk-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .graph-walk-list li {
            font-size: 0.85rem;
            font-family: monospace;
            padding-left: 1.25rem;
            position: relative;
            color: var(--text-muted);
        }
        .graph-walk-list li::before {
            content: '➔';
            position: absolute;
            left: 0;
            color: var(--accent-secondary);
        }
        pre.raw-prompt {
            font-family: monospace;
            font-size: 0.75rem;
            background: #020617;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow-x: auto;
            white-space: pre-wrap;
            color: #a7f3d0;
            line-height: 1.4;
        }
        .citation-tag {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            font-weight: 500;
            padding: 0.15rem 0.4rem;
            border-radius: 6px;
            font-size: 0.8rem;
            margin-top: 0.5rem;
            display: inline-block;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <header>
        <h1>Physics Hybrid RAG Tutor</h1>
        <div class="badge">Hybrid Retrieval: Vector + Keyword + Graph RRF</div>
    </header>

    <div class="main-layout">
        <!-- Chatbot Section -->
        <div class="chat-section">
            <div class="messages-container" id="chat-messages">
                <div class="message-bubble bot">
                    Hello Krishna! I am your NCERT Physics AI Tutor. I can answer conceptual, factual, or formula-based questions from Chapter 1 (Charges & Fields), Chapter 2 (Potential & Capacitance), Chapter 3 (Electricity), Chapter 4 (Magnetism), and Chapter 6 (Electromagnetic Induction).
                    <br><br>
                    Ask me anything! For example: <i>"What is Gauss Law?"</i> or <i>"Explain Electromagnetic Induction and state its formula."</i>
                </div>
            </div>
            <div class="input-bar">
                <input type="text" id="query-input" placeholder="Type your physics question here..." onkeypress="handleKeyPress(event)">
                <button class="send-btn" onclick="submitQuery()">Ask Tutor</button>
            </div>
        </div>

        <!-- Observability Dashboard Section -->
        <div class="dashboard-section">
            <div class="tab-bar">
                <button class="tab-btn active" onclick="switchTab(0)">Vector (Semantic)</button>
                <button class="tab-btn" onclick="switchTab(1)">BM25 (Keyword)</button>
                <button class="tab-btn" onclick="switchTab(2)">Knowledge Graph</button>
                <button class="tab-btn" onclick="switchTab(3)">Merged RRF Prompt</button>
            </div>
            
            <!-- Vector Search Tab -->
            <div class="tab-content active" id="tab-0">
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">
                    Semantic Vector similarity matching using FAISS / NumPy Cosine distance metrics.
                </p>
                <div id="vector-results">
                    <p style="color: var(--text-muted); font-style: italic; font-size: 0.85rem;">No search run yet.</p>
                </div>
            </div>
            
            <!-- BM25 Tab -->
            <div class="tab-content" id="tab-1">
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">
                    Lexical Term Frequency-Inverse Document Frequency matching using the BM25 formula.
                </p>
                <div id="bm25-results">
                    <p style="color: var(--text-muted); font-style: italic; font-size: 0.85rem;">No search run yet.</p>
                </div>
            </div>
            
            <!-- Knowledge Graph Tab -->
            <div class="tab-content" id="tab-2">
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">
                    Query expansion walks on topics, formulas, chapters, and sections represented inside NetworkX / CustomGraphDB.
                </p>
                <div id="graph-results">
                    <p style="color: var(--text-muted); font-style: italic; font-size: 0.85rem;">No search run yet.</p>
                </div>
            </div>
            
            <!-- RRF Prompt Tab -->
            <div class="tab-content" id="tab-3">
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">
                    The exact text prompt constructed using top fused RRF chunks and graph formulas, sent to the LLM API.
                </p>
                <pre class="raw-prompt" id="prompt-display">No search run yet.</pre>
            </div>
        </div>
    </div>

    <script>
        function handleKeyPress(e) {
            if (e.key === 'Enter') {
                submitQuery();
            }
        }
        
        function switchTab(idx) {
            const tabs = document.querySelectorAll('.tab-btn');
            const contents = document.querySelectorAll('.tab-content');
            
            tabs.forEach((tab, i) => {
                if (i === idx) tab.classList.add('active');
                else tab.classList.remove('active');
            });
            
            contents.forEach((content, i) => {
                if (i === idx) content.classList.add('active');
                else content.classList.remove('active');
            });
        }
        
        async function submitQuery() {
            const input = document.getElementById('query-input');
            const query = input.value.trim();
            if (!query) return;
            
            // Clear input
            input.value = '';
            
            // Append User message
            const container = document.getElementById('chat-messages');
            const userBubble = document.createElement('div');
            userBubble.className = 'message-bubble user';
            userBubble.innerText = query;
            container.appendChild(userBubble);
            container.scrollTop = container.scrollHeight;
            
            // Append Bot loader bubble
            const botBubble = document.createElement('div');
            botBubble.className = 'message-bubble bot';
            botBubble.innerText = 'Consulting hybrid indexes...';
            container.appendChild(botBubble);
            container.scrollTop = container.scrollHeight;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    const answerText = data.answer || "No response generated by the tutor.";
                    
                    // Render cited answer (replacing loader)
                    botBubble.innerHTML = formatMarkdown(answerText);
                    container.scrollTop = container.scrollHeight;
                    
                    // Render Observability diagnostics
                    renderDiagnostics(data.diagnostics);
                    
                    // Re-run MathJax formulas formatting (safely guarded against CDN timeouts/load states)
                    if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
                        try {
                            window.MathJax.typesetPromise([botBubble]).catch(err => console.log("MathJax format error:", err));
                        } catch (mathJaxErr) {
                            console.warn("MathJax typeset exception: ", mathJaxErr);
                        }
                    }
                } else {
                    botBubble.innerText = "Error contacting tutor server.";
                }
            } catch (err) {
                botBubble.innerText = "Connection failed.";
                console.error("UI Query execution failed: ", err);
            }
        }
        
        function formatMarkdown(text) {
            // Basic replacement for code tags or citations to bold them nicely
            let html = text.replace(/\\\\n/g, '<br>');
            html = html.replace(/\\n/g, '<br>');
            
            // Highlight citations like [Page 9, Chapter 1...]
            html = html.replace(/(\\[Page \\d+[^\\]]*\\])/g, '<span class="citation-tag">$1</span>');
            
            return html;
        }
        
        function renderDiagnostics(diag) {
            // 1. Render Vector search tab
            const vecDiv = document.getElementById('vector-results');
            if (diag.vector_search && diag.vector_search.length > 0) {
                vecDiv.innerHTML = diag.vector_search.map((item, idx) => `
                    <div class="diag-item">
                        <h4>Rank ${idx+1} <span class="score">Cos: ${item.score.toFixed(4)}</span></h4>
                        <p><strong>Page ${item.page}:</strong> ${item.chapter} - ${item.section}</p>
                    </div>
                `).join('');
            } else {
                vecDiv.innerHTML = '<p style="color: var(--text-muted); font-style: italic; font-size: 0.85rem;">No vector results matched.</p>';
            }
            
            // 2. Render BM25 search tab
            const bmDiv = document.getElementById('bm25-results');
            if (diag.bm25_search && diag.bm25_search.length > 0) {
                bmDiv.innerHTML = diag.bm25_search.map((item, idx) => `
                    <div class="diag-item">
                        <h4>Rank ${idx+1} <span class="score">BM25: ${item.score.toFixed(2)}</span></h4>
                        <p><strong>Page ${item.page}:</strong> ${item.chapter} - ${item.section}</p>
                    </div>
                `).join('');
            } else {
                bmDiv.innerHTML = '<p style="color: var(--text-muted); font-style: italic; font-size: 0.85rem;">No keyword results matched.</p>';
            }
            
            // 3. Render Knowledge Graph Tab
            const graphDiv = document.getElementById('graph-results');
            if (diag.graph_traversal && diag.graph_traversal.steps.length > 0) {
                let html = '<h4>Traversal Steps:</h4>';
                html += '<ul class="graph-walk-list" style="margin-top: 0.5rem; margin-bottom: 1.5rem;">';
                html += diag.graph_traversal.steps.map(step => `<li>${step}</li>`).join('');
                html += '</ul>';
                
                if (diag.graph_traversal.formulas.length > 0) {
                    html += '<h4>Injected Formulas:</h4>';
                    html += diag.graph_traversal.formulas.map(f => `
                        <div class="diag-item" style="margin-top: 0.5rem; border-color: rgba(168, 85, 247, 0.3);">
                            <p style="color: #c084fc; font-family: monospace;">${f}</p>
                        </div>
                    `).join('');
                }
                
                graphDiv.innerHTML = html;
            } else {
                graphDiv.innerHTML = '<p style="color: var(--text-muted); font-style: italic; font-size: 0.85rem;">No graph entities matched the query terms.</p>';
            }
            
            // 4. Render Merged RRF Prompt
            const promptPre = document.getElementById('prompt-display');
            // Reconstruct a visual mock of the exact prompt
            const chunksText = diag.rrf_merged.map((ch, idx) => 
                `Source ${idx+1} (Page ${ch.page}, ${ch.chapter}, ${ch.section}) [RRF: ${ch.rrf_score.toFixed(4)}, Vec Rank: ${ch.vector_rank || 'N/A'}, BM25 Rank: ${ch.bm25_rank || 'N/A'}]`
            ).join('\\n\\n');
            
            const formulasText = diag.graph_traversal.formulas.map(f => `  - ${f}`).join('\\n');
            
            promptPre.innerText = `[SYSTEM INSTRUCTIONS]\nYou are a helpful and precise Physics AI Tutor for NCERT Class 12 Physics...\n\n[KNOWLEDGE GRAPH FORMULAS]\n${formulasText}\n\n[SOURCE MATERIALS]\n${chunksText}\n\n[STUDENT QUESTION]\n${diag.query}`;
        }
    </script>
</body>
</html>
"""

class RAGHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging spam of chat queries
        if args and isinstance(args[0], str) and "POST /chat" in args[0]:
            return
        super().log_message(format, *args)
        
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_error(404, "File Not Found")
            
    def do_POST(self):
        if self.path == "/chat":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                req_json = json.loads(post_data.decode("utf-8"))
                query = req_json.get('query', '')
                
                # Execute Hybrid Search Retrieval
                retrieved_chunks, graph_formulas, diagnostics = ranker.search(query, k=3)
                
                # Generate answer grounded in context
                answer = chain.generate_answer(query, retrieved_chunks, graph_formulas)
                
                # Format Response
                response_data = json.dumps({
                    "answer": answer,
                    "diagnostics": diagnostics
                })
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_data.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint Not Found")

def start_server(port=8080):
    global ranker, chain
    print("Step 1: Reading NCERT Physics textbook chapters...")
    chunks = get_chunks()
    
    print("Step 2: Constructing vector database index...")
    vm = VectorStoreManager()
    vm.build_index(chunks)
    
    print("Step 3: Constructing NetworkX Physics Knowledge Graph...")
    kg = build_physics_knowledge_graph()
    
    print("Step 4: Initializing Hybrid Ranker & LLM Chain...")
    ranker = HybridSearchRanker(chunks, vm, kg)
    chain = RAGChain() # Will read environment API key or run in mock mode
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, RAGHTTPRequestHandler)
    print(f"\nPhysics Tutor RAG Server running on: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Physics Tutor RAG Web Application")
    parser.add_argument('--port', type=int, default=8080, help="Port number")
    args = parser.parse_args()
    
    # Configure path references
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    start_server(args.port)
