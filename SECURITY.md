# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's private vulnerability reporting
(Security tab → "Report a vulnerability"). You will get a response within 7 days.

## Scope

- Loci's Swift code and its handling of model files and downloads.
- Vulnerabilities in llama.cpp itself should be reported upstream; we will bump the pinned version once fixed.

## Model files

GGUF files are parsed by llama.cpp. Only load models from sources you trust, and
verify checksums (LociHub does this automatically when a checksum is known).

## Data handling

Loci is designed for sensitive documents. Reports of document text, queries or answers
leaking into logs, caches, temp files or unprotected files are treated as vulnerabilities.
