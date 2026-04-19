# Active Work

## [EVAL] Lock test indices + noise seeds (do first)
- Write `data/test_indices.npy` with 500 held-out indices
- Create `src/eval/constants.py` defining:
  - `TEST_INDICES_PATH = "data/test_indices.npy"`
  - `PAPER_SEEDS = [0, 1, 2]`
  - `N_TEST_SAMPLES = 500`
  - `TEACHER_DDIM_STEPS = 250`
- All new eval/regeneration scripts import from here
- ~10 min of work, one-time

## [INFRA] Fix `scripts/precompute_teacher_moments.py`
- Bump `DDIM_STEPS = 50` → import from `src/eval/constants.py` (= 250)
- Use locked test indices + seed from constants
- Skip `scripts/generate_presentation.py` — it's legacy slide-deck code, not paper-critical

## [MOMENT] Regenerate `teacher_moments.pt` on server
- Blocked on the above two
- Run: `python scripts/precompute_teacher_moments.py --gpu <X> --n_samples 1000`
- Output: `darcy_teacher/exp_1/saved_state/teacher_moments.pt` (overwrite)

## [MOMENT] Restart moment matching experiments on new moments
- Blocked on `teacher_moments.pt` regeneration
- Exp 18, 19, 20, 21, 22 — all currently on stale (50-step) moments
- Decide: restart from scratch, or resume from current epoch with new moments?

## [PAPER] Fill in `paper/main.tex` Methods section
- Methods section can be written now (doesn't depend on numbers)
- Experiments section stays as lipsum until unified eval results exist
