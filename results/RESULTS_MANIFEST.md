# Results Manifest

This manifest maps the retained paper and appendix figures/tables to their source artifacts, regeneration command, and output location.

| Item | LaTeX label / file | Source artifacts | Script / command | Output path |
| --- | --- | --- | --- | --- |
| Two-route overview figure | `fig:recoverability_loci` | manuscript source | manual LaTeX figure | `paper/figures/fig_recoverability_loci.tex` |
| Proposal-space pipeline figure | `fig:proposal_logic` | retained local figure asset | retained local copy | `paper/figures/fig_consolidation_logic.png` |
| Route / benchmark map table | `tab:route_protocol` | manuscript source plus retained benchmark roles | retained table source | `paper/tables/route_protocol.tex` |
| Four-route qualitative audit figure | `fig:route_audits` | rebuilt RWTD, ADE20K-selected, STLD, and ControlNet audit panels | `python scripts/build_neurips_explanatory_figures.py` | `paper/figures/fig_route_audits.png` |
| Main matched comparison table | `tab:proposal_results` | committed RWTD, STLD, ControlNet, and ADE20K-selected summaries under `results/artifacts/` | artifact-backed table source | `paper/tables/proposal_results.tex` |
| RWTD oracle decomposition figure | `fig:rwtd_oracle_decomp` | retained RWTD oracle and generic-baseline summaries | `python scripts/build_rwtd_oracle_figure.py` | `paper/figures/fig_rwtd_oracle_decomposition.png` |
| RWTD recoverability table | `tab:oracle_recoverability` | retained RWTD oracle and learned-selector summaries | retained table source | `paper/tables/oracle_recoverability.tex` |
| Proposal ablation table | `tab:proposal_ablation` | retained RWTD controls, STLD direct summary, and ControlNet summary | retained table source | `paper/tables/proposal_ablation.tex` |
| Decision-layer algorithm box | `tab:algorithm_box` | method in `paper/sections/03_problem_formulation.tex` and `paper/sections/05_proposal_space_recovery.tex` | manual manuscript summary | `paper/tables/algorithm_box.tex` |
| Descriptor ablation table | `tab:descriptor_ablation` | `results/artifacts/descriptor_ablation/descriptor_ablation_summary.csv` plus retained hybrid exports | `python scripts/build_descriptor_ablation.py --skip-rwtd-final --descriptor-modes handcrafted ptd_convnext hybrid_ptd --out-root results/artifacts/descriptor_ablation` | `paper/tables/descriptor_ablation.tex` |
| Practicality table | `tab:practicality` | retained profiling and audit artifacts documented in `results/UPDATED_SOURCES_USED.md` | retained table source | `paper/tables/practicality.tex` |
| Feature-space summary figure | `fig:feature_summary_app` | archived parity CSVs and retained gallery assets | `python paper/scripts/build_feature_summary_figure.py` | `paper/figures/fig_feature_space_recovery.png` |
| Feature parity table | `tab:feature_parity` | archived parity CSVs | retained table source | `paper/tables/feature_parity.tex` |
| Feature qualitative gallery | `fig:feature_gallery` | retained preview assets | `python paper/scripts/build_appendix_assets.py` | `paper/figures/fig_feature_gallery.png` |
| Feature scale diagnostics | `fig:feature_scale_diag` | archived scale-probe table | `python paper/scripts/build_appendix_assets.py` | `paper/figures/fig_feature_scale_diagnostics.png` |
| RWTD case gallery | `fig:rwtd_case_gallery` | retained oracle / selector CSVs and audit-case images | `python paper/scripts/build_appendix_assets.py` | `paper/figures/fig_rwtd_case_gallery.png` |
| RWTD ambiguity figure | `fig:ambiguity_commitment` | retained audit figure | retained local figure | `paper/figures/fig_ambiguity_commitment_full256.png` |
| STLD selector contrast table | `tab:stld_selector_contrast` | retained STLD learned-selector and oracle summaries | retained table source | `paper/tables/stld_selector_contrast.tex` |
| STLD oracle table | `tab:stld_oracles` | retained STLD oracle summaries | retained table source | `paper/tables/stld_oracles.tex` |
| STLD case gallery | `fig:stld_case_gallery` | STLD benchmark images / labels; final masks; selector masks; per-image oracle CSV | `python paper/scripts/build_appendix_assets.py` | `paper/figures/fig_stld_case_gallery.png` |
| ADE20K-selected audit figure | `fig:detexture_app` | retained ADE20K-selected evaluation bundle | `python scripts/build_detexture_audit_figure.py` | `paper/figures/fig_detexture_audit.png` |
| ADE20K-selected appendix table | `tab:detexture_secondary` | `results/artifacts/external_routes/detexture_ade20k/*` plus manuscript table source | retained table source | `paper/tables/detexture_secondary.tex` |
| ControlNet bridge appendix figure | `fig:controlnet_bridge_app` | retained ControlNet evaluation bundle | `python scripts/build_controlnet_audit_figure.py` | `paper/figures/fig_controlnet_bridge_examples.png` |
| ControlNet bridge appendix table | `tab:controlnet_secondary` | retained ControlNet summaries plus manuscript table source | retained table source | `paper/tables/controlnet_secondary.tex` |
| CAID appendix figure | `fig:caid_app` | retained CAID evaluation bundle | `python scripts/build_caid_audit_figure.py` | `paper/figures/fig_caid_audit.png` |
| CAID appendix table | `tab:caid_secondary` | retained CAID summaries plus manuscript table source | retained table source | `paper/tables/caid_secondary.tex` |

For end-to-end command provenance, use [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md). For the broader evidence map, use [UPDATED_SOURCES_USED.md](UPDATED_SOURCES_USED.md).
