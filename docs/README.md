# Building the RivGraph docs

From the repository root:

```bash
conda env create -f environment.yml
conda activate rivgraph-modern
pip install -e . --no-deps
pip install -e ".[docs]"
python -m sphinx -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser.

Optional convenience commands:

- macOS / Linux: `cd docs && make html`
- Windows: `cd docs && make.bat html`
- Link check: `python -m sphinx -b linkcheck docs/source docs/build/linkcheck`

Notes:

- `environment.yml` is the canonical source/development environment.
- `environment-modern.yml` is a transition alias and should match it.
- Generated docs should live under `docs/build/` and should not be committed.
