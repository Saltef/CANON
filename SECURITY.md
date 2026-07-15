# Security Policy

CANON is designed for local and internal evidence workflows over private
corpora. Treat mounted drives, local repositories, generated reports, and cached
corpus artifacts as sensitive unless you have reviewed their contents.

## Supported Versions

CANON is currently an alpha workbench. Security fixes are applied to the active
`master` branch until a versioned release policy is published.

## Reporting A Security Issue

Do not open public issues that include secrets, private corpus text, API keys,
credentials, personal data, or exploit details.

For now, report suspected security issues privately to the repository owner and
include only the minimum reproduction details needed to understand the issue.
When a private disclosure channel is added to the public repository, this file
should be updated before any broad package release.

## Secret Handling

- Keep API keys in `.env`; never commit real keys.
- Do not paste API keys, private documents, or sensitive corpus excerpts into
  prompts, issue reports, tests, generated docs, or committed fixtures.
- Review generated files under `reports/` before sharing them outside your
  organization.
- Mount Google Drive or other synced folders read-only when possible.

## Corpus Safety

CANON skips common repository internals, dependency folders, build outputs,
caches, and virtual environments by default. You are still responsible for
choosing safe source boundaries before ingestion.

Use `canon.product.mounted_corpus --profile-only` or the `/v1/sources/profile`
API before indexing private folders.
