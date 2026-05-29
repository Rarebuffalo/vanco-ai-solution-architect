# Vanco AI Solution Architect Technical Assessment

This repository contains the complete production-grade implementation of the three assessment projects. The solutions are designed around clean, self-contained architecture pipelines, rigorous validation discipline, and detailed observability rather than simple api-wrappers.

---

## Repository Structure

```
vanco-ai-solution-architect/
├── pyproject.toml                     # Unified package metadata & dependencies
├── .gitignore                         # Git exclusion rules for large datasets & venv
├── README.md                          # Global reproduction instructions (This file)
├── solution_report.md                 # Architect evaluation report (Approach, Metrics & Tradeoffs)
├── diagrams/                          # Pipeline and system architecture diagrams
│   ├── forecasting_architecture.svg   # Use Case 1 diagram
│   ├── asl_architecture.svg           # Use Case 2 diagram
│   └── hybrid_rag_architecture.svg    # Use Case 3 diagram
├── docs/                              # Developer guides and submission assets
│   ├── forecasting_guide.md           # Tabular modeling & temporal lag strategy guide
│   ├── asl_detection_guide.md         # Computer vision guide (dataset split, YOLO math)
│   ├── hybrid_rag_guide.md            # RAG design guide (BM25, Graph search, RRF rank)
│   └── kaggle_leaderboard.png         # Screenshot of public leaderboard score (0.43179)
├── grocery_forecasting/               # Use Case 1: Grocery Sales Forecasting
│   ├── data/                          # Data store (stores, oil, holidays, train, test)
│   ├── notebooks/
│   │   ├── 01_eda_and_features.ipynb  # Explanatory Data Analysis & lag structures
│   │   └── 02_training_eval.ipynb     # Model training & detailed error residuals
│   └── src/
│       ├── features.py                # Feature pipeline (lags, rolling averages, events)
│       ├── validation.py              # Chronological train/validation time-splitter
│       ├── model.py                   # LightGBM training & subpopulation analysis
│       └── test_run.py                # End-to-end pipeline test runner (synthetic data)
├── asl_detection/                     # Use Case 2: ASL Hand Sign Object Detector
│   ├── data/                          # Image repository & YOLO dataset config
│   └── src/
│       ├── capture.py                 # Active webcam capture & synthetic dataset generator
│       ├── train.py                   # YOLOv8 trainer with offline emulation fallback
│       ├── demo.py                    # OpenCV-based local webcam demo loop
│       └── server.py                  # Standalone HTTP POST fallback server for remote reviews
└── hybrid_rag/                        # Use Case 3: Hybrid NCERT Physics RAG Tutor
    ├── data/                          # NCERT Physics PDF & serialization stores
    └── src/
        ├── ingest.py                  # Section-aware PDF parser with page/formula trackers
        ├── vector_db.py               # Vector index manager (FAISS & NumPy TF-IDF fallbacks)
        ├── graph_db.py                # Adjacency-list Knowledge Graph (NetworkX fallback)
        ├── search.py                  # BM25 Keyword + Semantic + Graph Hybrid Ranker (RRF)
        ├── rag_chain.py               # Grounded prompt compiler & Gemini HTTP poster
        └── server.py                  # Built-in Web Server (Chat UI & Observability Dashboard)
```

---

## Installation & Setup

All projects are Python-based and use standard scientific computing and web libraries. We recommend using the fast package manager `uv` or standard virtual environments.

### Option A: Setup using `uv` (Recommended)
```bash
# 1. Create a virtual environment
uv venv

# 2. Activate virtual environment
source .venv/bin/activate  # On Linux/macOS

# 3. Install all dependencies defined in pyproject.toml
uv pip install -e .
```

### Option B: Setup using standard `pip`
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -e .
```

*Note: If you run into compilation errors (e.g. FAISS or OpenCV native bindings on headless nodes), the code contains native Python/NumPy fallback wrappers that automatically run the entire pipeline offline.*

---

## Use Case 1: Grocery Sales Forecasting

We implement a time-series regression pipeline using LightGBM. It prevents time leakage using a chronological validation window and uses out-of-sample lag shifts.

### Empirical Performance
* **Local Validation RMSLE**: **`0.40718`**
* **Kaggle Public Leaderboard RMSLE**: **`0.43179`**
* **Kaggle Submission Evidence**: Proof screenshot is saved at [docs/kaggle_leaderboard.png](file:///home/Krishna-Singh/vanco-ai-solution-architect/docs/kaggle_leaderboard.png).

### Running End-to-End Test (Synthetic Data)
To verify that the feature engineering, time splitting, LightGBM model, error analysis, and prediction files execute correctly on your machine, run our pipeline test script:
```bash
PYTHONPATH=grocery_forecasting/src python grocery_forecasting/src/test_run.py
```
This generates synthetic tables in `grocery_forecasting/data/` and trains a baseline model, printing subpopulation error residuals (family, stores, holidays) to the terminal.

### Production Run with Kaggle CSVs
1. Place the Kaggle competition tables (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`) in `grocery_forecasting/data/`. If you downloaded the zipped versions:
   ```bash
   unzip -q train.csv.zip -d grocery_forecasting/data/
   unzip -q transactions.csv.zip -d grocery_forecasting/data/
   ```
2. Run the pipeline:
   ```bash
   PYTHONPATH=grocery_forecasting/src python grocery_forecasting/src/model.py \
     --train_path grocery_forecasting/data/train.csv \
     --test_path grocery_forecasting/data/test.csv \
     --stores_path grocery_forecasting/data/stores.csv \
     --oil_path grocery_forecasting/data/oil.csv \
     --holidays_path grocery_forecasting/data/holidays_events.csv \
     --output_path grocery_forecasting/data/submission.csv
   ```
3. Submissions will be exported to `grocery_forecasting/data/submission.csv`.


---

## Use Case 2: American Sign Language Detection

A computer vision pipeline utilizing YOLOv8 for hand object detection and classification. Supports local camera execution or remote browser streaming.

### Step 1: Prepare the Dataset
If you do not have a pre-annotated dataset, you can instantly construct a sample dataset containing 8 classes (A to H) and 20 images per class in YOLO format:
```bash
python asl_detection/src/capture.py --mode synthetic
```
This generates 160 annotated hand images using skin-colored polygons in `asl_detection/data/dataset/` ready for immediate model training.
*(To capture your own images via webcam, run: `python asl_detection/src/capture.py --mode capture`)*

### Step 2: Train the YOLOv8 Model
```bash
python asl_detection/src/train.py --epochs 15 --batch 8
```
This trains a YOLOv8 Nano model on the dataset and exports validation curves, a confusion matrix, and best weights (`best.pt`) to the `runs/detect/train/` folder.

### Step 3: Run the Live Demo
We provide two live demo entrypoints depending on your machine environment:

#### Mode A: Local Desktop (Natively runs on a laptop with camera)
```bash
python asl_detection/src/demo.py
```
This opens a standard OpenCV `cv2.imshow` window, capturing live camera frames, executing YOLOv8 predictions on CPU, and overlaying bounding boxes in real-time.

#### Mode B: Headless Remote Server (Runs inside containers or VM hosts)
If you are reviewing the code on a remote server with no display window or mapped webcam, start our built-in web server:
```bash
python asl_detection/src/server.py --port 8000
```
Then, open **`http://localhost:8000`** in your browser. The browser webpage will capture frames from your local webcam using HTML5 `getUserMedia`, POST them to the server at 10 FPS, receive bounding boxes, and draw them on a canvas.

---

## Use Case 3: Hybrid Physics RAG System

An educational AI Tutor parsing NCERT Physics, indexing it across FAISS Vector DB, BM25 Keyword Index, and NetworkX Graph DB, and outputting responses with verified page and section citations.

### Step 1: Start the RAG Web Application
Start our self-contained web server:
```bash
PYTHONPATH=hybrid_rag/src python hybrid_rag/src/server.py --port 8080
```
This starts the ingestion pipeline (which automatically downloads NCERT chapters or falls back to our local physics database if offline), builds the vector index, populates the NetworkX knowledge graph of concepts/formulas, and launches the HTTP service.

### Step 2: Open the Chatbot & Observability Dashboard
Navigate to **`http://localhost:8080`** in your browser.

* **Left Column (AI Tutor Chat)**: Interactive chatbot. Try asking:
  * *"What is Gauss Law?"*
  * *"Explain Electromagnetic Induction and state its formula."*
  * *"Compare Electric Flux and Electric Field."*
  * *"What is the capital of France?"* (Refusal test: the system must refuse to answer as it is not in the textbook).
* **Right Column (Observability Dashboard)**: Real-time diagnostic panel. Click on tabs to inspect:
  * **Vector (Semantic)**: Consine similarity scores and page source snippets.
  * **BM25 (Keyword)**: BM25 score calculations for terms.
  * **Knowledge Graph**: traversals (e.g. Coulomb's Law Topic -> Coulomb Formula -> Page 9).
  * **Merged RRF Prompt**: The exact compiled prompt context sent to the LLM.

*Note: The LLM generation connects to the Gemini API if `GEMINI_API_KEY` is present in your environment variables. If absent, the system runs in an offline mock generator mode, using the retrieved context to provide cited physics answers, making it fully testable without API tokens.*
