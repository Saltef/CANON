# Phase 15: Static Workbench

## Goal

Make experiment outputs inspectable without requiring a frontend stack.

## Implementation

The workbench generator reads existing reports and writes a static HTML page with
policy summaries, query answers, claim/conflict counts, and run metadata.

## Outputs

- `reports/workbench_<mode>.html`
- `reports/workbench_<mode>.json`
