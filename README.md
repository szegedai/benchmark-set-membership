# Hairline Cracks in Simple Problems: Benchmarking the Robustness of LLMs on Set Membership

A benchmark for evaluating how accurately LLMs solve the **set membership** task — i.e., whether a given element belongs to a set. The benchmark tests models across a range of prompt styles (natural language and code) and set types (related/unrelated words and numbers), with systematic variation of features.

---

## Overview

The core task: given a set of elements and a query element, does the query element belong to the set? Despite its simplicity, models' ability to solve it correctly can vary considerably depending on how the prompt is framed.

The benchmark explores two key dimensions:

- **Prompt style** — natural language vs. Python code (simple and algorithmic variants)
- **Feature variation** — quotation style, listing format, variable names, argument order, logic structure, indentation, etc.

Each combination of features generates a unique prompt, and every prompt is evaluated across all permutations of each set (both positive and negative examples, plus intruder-word cases).

---

## Project Structure

```
code/
├── 00_test__construct_prompts.ipynb   # Notebook for inspecting/testing prompt construction
├── 01_evaluate__prompts.py            # Main evaluation script
├── 01_evaluate__prompts.sh            # Shell script to run evaluation across all models/sets/settings
├── 02_evaluate__answers.ipynb         # Notebook for analyzing model outputs
├── data/
│   └── config/
│       ├── sets.yaml                          # Set definitions (elements + intruder words)
│       ├── prompts_natural_1_2.yaml           # Natural language prompt templates (NL1)
│       ├── prompts_natural_3_4.yaml           # Natural language prompt templates (NL2)
│       ├── prompts_coding_simple.yaml         # Simple code prompt templates (CS)
│       └── prompts_coding_algorithmic.yaml    # Algorithmic code prompt templates (CA)
├── src/
│   ├── model.py      # Model loading and inference utilities
│   ├── prompts.py    # Prompt construction logic
│   └── sets.py       # Set permutation generation
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. The evaluation script defaults to `cuda:0`.

---

## Usage

### Running the evaluation

```bash
python 01_evaluate__prompts.py \
    --setting <setting> \
    --model <model_path> \
    --set <set_name> \
    [--gpu <gpu_id>] \
    [--seed <seed>] \
    [--force]
```

**Arguments:**

| Argument | Description |
|---|---|
| `--setting` | Prompt style: `natural_1_2`, `natural_3_4`, `coding_simple`, `coding_algorithmic` |
| `--model` | HuggingFace model path (e.g., `meta-llama/Llama-3.2-3B-Instruct`) |
| `--set` | Name of the set to evaluate (see `data/config/sets.yaml`) |
| `--gpu` | GPU index (default: `0`) |
| `--force` | Re-run evaluation even if output already exists |

Results are saved to `data/results/results_<setting>/<model>/<set>/`.

### Running all experiments

To evaluate all models, settings, and sets at once:

```bash
bash 01_evaluate__prompts.sh
```

---

## Sets

Sets are defined in `data/config/sets.yaml`. Each set has a list of `elements` and an optional `intruder` word (an element that superficially fits the context but does not belong to the set).

| Category | Examples |
|---|---|
| **Strongly related words (complete)** | cardinal directions, Harry Potter houses, DNA bases, math operations, card suits |
| **Loosely related words** | animals, fruits, clothes |
| **Unrelated words** | random word mixtures |
| **Related numbers** | arithmetic sequences (step 2, 75, 100) |
| **Unrelated numbers** | random number mixtures |

---

## Prompt Settings

| Setting | Description |
|---|---|
| `natural_1_2` | Natural language prompts (NL1) |
| `natural_3_4` | Natural language prompts (NL2) |
| `coding_simple` | Python code prompt with inline `in` operator (CS) |
| `coding_algorithmic` | Python code prompt with function implementation (CA) |

Within each setting, prompts vary across multiple surface features such as quotation style, listing format (inline vs. bullet points), variable names, argument order, indentation style, and presence of type hints or docstrings.

---

## Models Evaluated

The shell script (`01_evaluate__prompts.sh`) is pre-configured for the following models:

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-Small-24B-Instruct-2501`
- `Qwen/Qwen2.5-32B-Instruct`
- `microsoft/Phi-3.5-MoE-instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`

---

## Output Format

Each result file is a JSON array. Each entry corresponds to one prompt evaluation:

```json
{
    "prompt": "...",
    "answer": "...",
    "features": {"init": 1, "quotation": 2, ...},
    "elements": ["West", "North", "East"],
    "element": "South",
    "groundtruth": true,
    "has_intruder": false
}
```

---

## Analysis

Use `02_evaluate__answers.ipynb` to load and analyze results. It processes model answers, compares them against ground truth, and aggregates accuracy across feature combinations and set types.
