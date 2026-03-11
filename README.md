# CiteSelect

Semantic Scholar citation mining pipeline for a target researcher.

It fetches the researcher's papers, collects all citing papers and citation contexts,
uses LLM-based refinement to rank representative citations, enriches author identity
signals with academic-awards datasets, and exports a reviewable Excel workbook for
grant / talent / evaluation materials.

## What It Produces

- `outputs/citations/`: raw citing papers and citation contexts
- `outputs/citations_refined/`: LLM-refined citation judgments
- `outputs/reports/representative_citations.xlsx`: final Excel export

Main Excel columns:

- `citing paper title`
- `citing venue`
- `authors`
- `author type`
- `citation role`
- `importance score`
- `citing content`

## Repo Layout

- `collect.py`: CLI entrypoint
- `fetch_citations.py`: fetch citing papers from Semantic Scholar
- `refine_citations.py`: compact LLM refinement
- `export_excel.py`: Excel export and post-filters
- `award_matcher.py`: local fuzzy matching against award/fellow datasets
- `academic-awards/data/`: vendored award/member datasets
- `config.example.yaml`: example config without secrets

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Create a local config from the example:

```bash
cp config.example.yaml config.yaml
```

Fill at least:

- `author_url` or `author_id`
- `s2_api_key` which is the Semantic Scholar API key
- `openrouter_api_key` if using the default OpenRouter setup

Notes:

- `config.yaml` is intentionally ignored by git because it may contain secrets.
- `academic-awards/data/` is committed and used for local fellow/member matching.

### How to get `s2_api_key`

`s2_api_key` means the Semantic Scholar API key.

Official page:

- https://www.semanticscholar.org/product/api

Request an API key there, then put it into `config.yaml`:

```yaml
s2_api_key: "YOUR_SEMANTIC_SCHOLAR_API_KEY"
```

### How to get `author_url` and `author_id`

Open the target researcher’s Semantic Scholar author profile page and copy the URL.

Example:

```text
https://www.semanticscholar.org/author/Yu-Huan-Wu/48607882
```

In this example:

- `author_url` is the full URL
- `author_id` is the numeric suffix at the end: `48607882`

This repository can parse `author_id` automatically from `author_url`, so in most cases setting `author_url` alone is enough.

Official references:

- Graph API docs: https://api.semanticscholar.org/api-docs/graph
- Semantic Scholar API / FAQ page: https://www.semanticscholar.org/faq#api-key-form

## Usage

Fetch the target researcher's publication list:

```bash
python3 collect.py fetch-publications
```

Fetch all citations for all target papers:

```bash
python3 collect.py fetch-citations
```

Run LLM refinement on all fetched citations:

```bash
python3 collect.py refine-citations --no-echo-llm
```

Export the final Excel workbook:

```bash
python3 collect.py export-excel
```

Run the full pipeline:

```bash
python3 collect.py all
```

Useful debug commands:

```bash
python3 collect.py fetch-citations --first-n 1 --per-paper-max 100
python3 collect.py refine-citations --first-n 1 --no-echo-llm
python3 collect.py export-excel --first-n 1
```

## Matching and Filtering

### Author Type

`author type` is built from:

- local award/member matching in `academic-awards/data/`
- optional manual additions in `author_profiles.yaml`
- citing-paper citation count heuristics

The author matcher is conservative by design:

- rule-based exact / near-exact matching first
- optional LLM filter removes suspicious tags using only `authors` and `author type`

### Venue Type

`venue type` is assigned by:

- local rules for common CV / AI / medical-imaging venues
- optional LLM correction for uncertain venues

Categories:

- `顶级期刊/会议`
- `高水平期刊/会议`
- `主流期刊/会议`
- `预印本`
- `其他期刊/会议`

## Outputs and Git

- All runtime outputs go under `outputs/`
- `outputs/` is ignored by git
- only source code, config examples, and necessary static datasets are intended to be committed

## Data Source Note

The `academic-awards` folder includes vendored award/member datasets and upstream scraping scripts.
Its nested `.git` metadata is ignored so the files can live inside this repository as normal tracked files.

## Acknowledgments

- Academic award / fellow datasets are based on `academic-awards` by xiaohk:
  https://github.com/xiaohk/academic-awards
- Citation graph, author profiles, and citing-context retrieval rely on Semantic Scholar:
  https://www.semanticscholar.org/
