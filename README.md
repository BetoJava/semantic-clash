# Semantic clash
## Folder structure


## Setup
```bash
pip install -r requirements.txt

# Default embedding model setup (CPU)
python setup.py

# OR for GPU
python setup.py --device cuda
```

## Preprocess
```bash
python main.py --mode preprocess --input input/document.docx --output data/ --chunking linebreak --device cuda
```

## Process
```bash
python main.py --mode process-inter-h2 --input data/document_vector_db --output data/inter_h2_reports/ --top-k 100 --min-similarity 0.0 --exclude-threshold 0.999 --score-threshold 0.5 --score-exponent 2.0
```

