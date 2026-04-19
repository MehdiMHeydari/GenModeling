# Active Work

## [PAPER] Generate visual sample figures
- `scripts/generate_paper_figures.py` produces `sample_grid.png` + `histograms.png`
- Next step after initial numeric eval
- Run: `python scripts/generate_paper_figures.py --gpu <X>`

## [PAPER] Fill in `paper/main.tex` Methods section
- Methods section can be written now (doesn't depend on numbers)
- Use content from `EXPERIMENT_LOG.md` + session discussion for each subsection
- Experiments section stays as lipsum until all numeric results + figures are finalized

## [EVAL] Decide whether to re-train MM on fresh teacher moments
- The evaluated MM-exp20/21/22 checkpoints were **trained against stale 50-step teacher moments**.
  `teacher_moments.pt` has since been regenerated with 250-step samples.
- Question: are the stale-moment MM checkpoints good enough, or do we restart training on fresh moments?
- Evidence for "good enough": MM-exp22 already gets WD 0.049 vs CD baseline 0.126 (2.6× improvement). Story is solid.
- Evidence for "restart": targets they're optimizing against aren't what we claim in the paper.
- Cost of restart: ~1 day per variant × 3 variants.
- Pragmatic option: re-evaluate current checkpoints on the fresh moments as a sanity check before deciding.
