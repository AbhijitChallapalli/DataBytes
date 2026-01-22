# DataBytes (MkDocs Blog)

This repo builds and publishes a MkDocs site to **GitHub Pages**.

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
mkdocs serve
```

Open: http://127.0.0.1:8000/

## Deploy

Push to `main` (or `master`). GitHub Actions will run and deploy to the `gh-pages` branch.

Then in your repo:
- **Settings → Pages → Source:** `Deploy from a branch`
- **Branch:** `gh-pages` / `(root)`

Your site will be available at:
`https://<username>.github.io/<repository>/`
