# Prompt Engineering for Named Entity Extraction from Portuguese Legal Documents

This repository contains the code and prompts used in the experiments described in the paper published at PROPOR 2026, conducted at the [Centre for Informatics and Systems of the University of Coimbra (CISUC)](https://www.cisuc.uc.pt/) and [Department of Informatics Engineering (DEI)](https://www.uc.pt/fctuc/dei/).

Giovanni Maffeo, Catarina Silva and Hugo Gonçalo Oliveira  
[Prompt Engineering for Named Entity Extraction from Portuguese Legal Documents](https://aclanthology.org/2026.propor-1.116/)  
[17th International Conference on Computational Processing of Portuguese (PROPOR 2026), Vol. 1, pages 1092-1097, Salvador, Brazil, 2026](https://propor2026.citius.gal/)

```
@inproceedings{maffeo2026prompt,
  title={Prompt Engineering for Named Entity Extraction from Portuguese Legal Documents},
  author={Maffeo, Giovanni and Silva, Catarina and Oliveira, Hugo Gon{\c{c}}alo},
  booktitle={Proceedings of the 17th International Conference on Computational Processing of Portuguese (PROPOR 2026)-Vol. 1},
  pages={1092--1097},
  year={2026}
}
```

Motivated by the limited availability and high cost of annotated legal data, which is a challenge that is even more severe for the Portuguese language, this work investigates whether prompt engineering over Large Language Models (LLMs) can effectively support legal Named Entity Recognition (NER) in low-supervision and low-resource settings through In-Context Learning (ICL). Using the [LeNER-Br corpus](https://github.com/peluz/lener-br), we evaluate category-specific prompts, different chunking sizes, and prompt engineering strategies. Entity-level evaluation using Exact Match Micro F1 shows that prompt engineering has a stronger impact on performance than other strategies.

We kindly request that users cite our paper in any publication that is generated as a result of the use of our source code or prompts.

## Requirements

1. [Python 3.6](https://www.python.org/downloads/)
2. [pip](https://pip.pypa.io/en/stable/installing/)

## Prompts

The [prompts/](prompts/) directory contains all prompt variations tested in our experiments for **Research Question 3 (RQ3)**, as detailed in section **2.2 Research Questions** of the paper:

- `baseline.txt`
- `baseline_definition_guideline.txt`
- `baseline_definition_guideline_examples.txt`
- `baseline_definition_guideline_erroranalysis.txt`
- `baseline_definition_guideline_examples_erroranalysis_1shot.txt`
- `baseline_definition_guideline_examples_erroranalysis_5shot.txt`

## Installation

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The main configuration is in [main.py](main.py). There are three execution modes:

### 1. Open-Source Models (default)
Set `API_MODE = False` and `OPENAI_MODE = False` to use local models via Ollama:
```python
API_MODE = False
OPENAI_MODE = False
MODEL_NAME = "qwen3:8b"  # or any Ollama model
```

### 2. OpenAI API
Set `OPENAI_MODE = True` and configure your API key in `.env`:
```python
OPENAI_MODE = True
```
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. External API
Set `API_MODE = True` to use an external endpoint serving a model. Configure the API endpoint and credentials in `.env`:
```python
API_MODE = True
OPENAI_MODE = False
```
Create a `.env` file with:
```
API_ENDPOINT=http://your-endpoint:8000/v1/chat/completions
API_PROJECT=your_project_name
```

Additional configuration options:
- `DATA_DIR` - Dataset split to use (default: `data/test`, options: `data/train`, `data/valid`, `data/test`)
- `PROMPT_NAME` - Select which prompt to use (matches filename in prompts/)
- `BY_ENTITY_MODE` - Enable entity-specific prompts
- `MAX_CHUNK_SIZE` - Maximum text chunk size for processing

## Execution

After configuration, execution is straightforward. Activate the virtual environment and run:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

The script will save results in [results/](results/) with a timestamped folder (`result_{timestamp}`) containing:
- `outputs/` - Model annotations output
- `prompts/` - Copy of prompt template used
- `log.txt` - Execution log
- `evaluation/` - Entity-level evaluation metrics

## Other Directories

### Results

The [results/](results/) directory contains detailed experimental results organized by research question, as described in section **3 Results** of the paper:

- `baseline/`
- `research_question_1/`
- `research_question_2a/`, `research_question_2b/`, `research_question_2c/`
- `research_question_3b/` through `research_question_3f/`
- `research_question_4a/` through `research_question_4f/`

### Error Distribution Analysis

The [error_distribution_analysis/](error_distribution_analysis/) directory contains the manual error analysis conducted on model predictions, including annotated chunks, entity categorization, and generated visualizations, as described in section **3 Results** of the paper.
