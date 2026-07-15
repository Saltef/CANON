# Supported Sources

CANON can ingest local files from normal folders, mounted Google Drive folders,
and local git repositories.

## Text-Extractable Files

These can become evidence text when their contents are readable:

- JSON, JSONL
- CSV
- TXT
- Markdown
- PDF
- DOCX
- XLSX
- PPTX
- HTML
- Jupyter notebooks
- common source-code files
- common config files such as YAML, TOML, XML, SQL, shell, and PowerShell files

## Git Repositories

Point CANON at a local checkout:

```powershell
python -m canon.product.mounted_corpus --input "C:\path\to\repo" --mode repo_review_v1 --profile-only
```

CANON indexes source and documentation files, including extensionless repository
files such as:

- `README`
- `LICENSE`
- `NOTICE`
- `CHANGELOG`
- `SECURITY`
- `Dockerfile`
- `Makefile`
- `Jenkinsfile`

It skips repository internals and noisy generated directories by default,
including `.git`, `node_modules`, `vendor`, `dist`, `build`, `target`,
`__pycache__`, `.pytest_cache`, `.venv`, and similar folders.

## Detected But Not Extracted By Default

These files are reported but not treated as evidence text unless converted or
OCR-extracted first:

- Google-native pointer files: `.gdoc`, `.gsheet`, `.gslides`
- images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`
- legacy or OpenDocument presentations: `.ppt`, `.odp`

Export Google-native files to PDF, DOCX, XLSX, PPTX, Markdown, HTML, CSV, or
plain text before using them as evidence.

## Security Boundary

CANON reads files from the path you provide. Keep the input folder curated:

- Do not include `.env` files.
- Do not include private keys, tokens, OAuth credentials, or cloud credentials.
- Do not include customer data or private material unless you intend to index it.
- Review generated `data/` and `reports/` artifacts before sharing them.
