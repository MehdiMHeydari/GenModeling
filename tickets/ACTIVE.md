# Active Work

## [EVAL] Re-run unified eval with structural diversity metric
- Running overnight on 1 GPU, single tmux session (eval_all)
- New metrics: `pix_diversity` + `struct_diversity` per row
- MM variants pinned to checkpoint_75.pt (struct-diversity peak)
- Output: `results/eval_all.csv`

## [PAPER] Generate visual sample figures with pinned MM checkpoints
- Needs to wait until eval finishes so we reuse the pinned checkpoints consistently
- Run: `python scripts/generate_paper_figures.py --gpu <X>`

## [INFRA] Kick off Navier-Stokes dataset work
- Confirm `data/2D_NS_incom_inhom_Re10000_128.h5` exists on server
- If missing: download from PDEBench
- Once data confirmed, train NS teacher (single GPU, ~1 day)
  - Can start in parallel with the current Darcy eval (teacher only uses 1 GPU)
- In parallel, train NS RF + Reflow + MFM (teacher-independent)

## [PAPER] Fill in `paper/main.tex` Methods section
- Methods section can be written now (doesn't depend on numbers)
- Use content from `EXPERIMENT_LOG.md` + session discussion for each subsection
- Experiments section stays as lipsum until Darcy + NS numbers are in

## [EVAL] Decide whether to re-train MM on fresh teacher moments
- Current MM variants were trained against stale 50-step teacher moments
- The struct-diversity peak at checkpoint_75 suggests training *is* working;
  the deeper issue is loss design (see BACKLOG: spatially-aware moment loss)
- Deferred — paper v1 can use checkpoint_75 MM results as-is
