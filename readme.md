## Setup

Run the environment file to install all required dependencies:
```bash
pip install -r environemnt.yml
```

## Usage

For each dataset, the pipeline consists of two steps:

**Step 1 — Snapshot Fine-tuning**
Run the training script to fine-tune the model for each snapshot:
```bash
python {dataset}_training.py 
```

**Step 2 — Snapshot Inference**
Run the inference script to generate alignment results for each snapshot:
```bash
python {dataset}_inference.py
```

Repeat Steps 1 and 2 sequentially for each snapshot.