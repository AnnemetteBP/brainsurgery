# Competing-tool result records

These compact records retain the reported-size Linux comparison. Reportable
results must pass every gate in `../protocol.md`; raw records stay below
`log/revision_tests/` and are not committed. The macOS machine-readable JSON is
retained only as harness history; no empty Mac paper table is retained.

## Linux reported-size result

- **Use [`paper_table.md`](paper_table.md) or
  [`paper_table.tex`](paper_table.tex) for the paper.** These are the canonical,
  populated fragments from the Linux run.
- [`linux_2dbcd50/`](linux_2dbcd50/paper_table.md): 30/30 independently validated measured outputs for run `eacl2027_competing_linux_2dbcd50` at commit `2dbcd505115100f892e906413076ae93b3fcaa16`.
- The stored `summary.json` is from the anonymized text export; the Markdown, LaTeX, and narrative fragments contain no machine-local paths or hostname.
- Raw evidence: `log/revision_tests/eacl2027_competing_linux_2dbcd50/competing_tools/`.
