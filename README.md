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

## Preprocessing
This will chunk, vectorise and store the data.
```bash
python main.py --mode preprocess --input input/document.docx --output data/
```

## Process
This will compute cos similarity distances and generate a semantic collision report.
```bash
# Génération du rapport
python main.py --mode process --input data/document_vector_db --output report/report.csv
```