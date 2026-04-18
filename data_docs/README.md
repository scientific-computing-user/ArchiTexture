# Dataset Notes

The final paper uses five benchmark routes with distinct roles:

| Role | Route | Readout emphasis |
| --- | --- | --- |
| historical natural anchor | `RWTD` | official invariant comparison plus oracle / selector analysis |
| natural complement | `ADE20K-selected` | visually selected public textural crops; invariant matched validation comparison |
| historical synthetic anchor | `STLD` | direct-foreground comparison; singleton-selection versus commitment contrast |
| synthetic complement | `ControlNet bridge` | invariant comparison on more naturalistic stitched transitions |
| appendix real-world complement | `CAID` | invariant shoreline / overhead-imagery breadth check |

This public repo does not redistribute the underlying benchmark image data. The retained result summaries, paper tables, and figure manifests document the exact subsets and evaluator conventions used in the manuscript.
