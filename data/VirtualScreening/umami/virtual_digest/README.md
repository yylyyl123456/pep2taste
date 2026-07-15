# Virtual Digestion Umami Analysis

This directory contains web-server-ready download files and summary tables for
the AnOxPP virtual hydrolysis peptide pool.

The model prediction was run once on the deduplicated union file:
`data/unreported_fragments_all.fasta`.

The nine AnOxPP method-specific files were used only to annotate peptide source
methods. Model prediction was not repeated for each method.

## Download tiers

- `lt_0.50`: Final_Prob < 0.50
- `gte_0.50`: Final_Prob >= 0.50
- `gte_0.90`: Final_Prob >= 0.90
- `gte_0.95`: Final_Prob >= 0.95

Each download file contains:

- sequence
- length
- hydrolysis method
- number of source methods producing this peptide
- ESM2/PepBERT/ProtT5 probabilities and labels
- weighted ensemble probability and label

Use `tables/web_download_manifest.csv` to populate web-server download links.
