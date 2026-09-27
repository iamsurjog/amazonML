# TechnoBladers

## Setup

### For Windows:

```pwsh
py -m venv ./.venv
.\.venv\bin\Activate
pip install -r requirements.txt
huggingface-cli download akshatbakshi/amazon-ml-challenge-2026 --repo-type dataset --local-dir ./data
```

### For Linux/Mac:

```pwsh
py -m venv ./.venv
source ./.venv/bin/activate
pip install -r requirements.txt
huggingface-cli download akshatbakshi/amazon-ml-challenge-2026 --repo-type dataset --local-dir ./data
```


### Dataset:

```py
from datasets import load_dataset

ds = load_dataset("akshatbakshi/amazon-ml-challenge-2026")
```


