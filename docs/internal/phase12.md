# Phase 12: Generation Providers

## Goal

Make answer generation swappable while keeping citations and evidence as the
stable contract.

## Implementation

The default backend is `grounded-template-v1`. An OpenAI backend is available
when `OPENAI_API_KEY` is set. Synthesis reports include generator provenance.

## Outputs

- synthesis reports include `generator.provider` and `generator.model`
- deterministic offline generation remains the default
