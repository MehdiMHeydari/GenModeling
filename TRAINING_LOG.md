# Training Run Log

## Teacher (VP Diffusion)
All teacher runs stored under `darcy_teacher/`

### darcy_teacher/exp_1/
- **Config:** batch_size=64, lr=1e-4 (constant), epochs=400
- **Schedule:** Cosine noise schedule, s=0.008
- **Architecture:** UNet, channels=64, channel_mult=1,2,4,4, res_blocks=2
- **Result:** Raw weights at epoch 200 chosen as best teacher checkpoint
- **Key finding:** EMA weights (0.9999) performed worse than raw weights across all epochs
- **Checkpoints:** saved_state/checkpoint_{0,25,50,...,399}.pt

---

## Student (Consistency Distillation)
All student runs stored under `darcy_student/`

### darcy_student/exp_1/
- **Config:** student_steps=4, batch_size=64, lr=1e-4, epochs=400
- **Teacher:** darcy_teacher/exp_1/saved_state/checkpoint_200.pt (raw weights)
- **CD params:** ema_rate=0.9999, x_var_frac=0.75, huber_epsilon=1e-4
- **N_teacher annealing:** 64->1280 over 100k iterations
- **Result:** Mode collapse - all samples are centered blobs (data mean)
- **Diagnosis:** student_steps=4 makes segments too large (25% of noise schedule each), consistency constraint too hard, model collapses to mean
- **Checkpoints:** saved_state/checkpoint_{0,25,50,75,100,...}.pt

### darcy_student/exp_2/
- **Config:** student_steps=8, batch_size=64, lr=1e-4, epochs=1000
- **Teacher:** darcy_teacher/exp_1/saved_state/checkpoint_200.pt (raw weights)
- **CD params:** ema_rate=0.9999, x_var_frac=0.75, huber_epsilon=1e-4
- **N_teacher annealing:** 64->1280 over 100k iterations
- **Changes from exp_1:** student_steps 4->8 (segments 12.5% instead of 25%), epochs 400->1000 (141k iterations, enough to complete annealing)
- **Result:** Diversity emerging, loss trending down (0.04→0.02). Much better than exp_1. Still training.
- **Checkpoints:** saved_state/checkpoint_{0,25,50,...}.pt

### darcy_student/exp_3/
- **Config:** student_steps=16, batch_size=64, lr=1e-4, epochs=1000
- **Teacher:** darcy_teacher/exp_1/saved_state/checkpoint_200.pt (raw weights)
- **CD params:** ema_rate=0.9999, x_var_frac=0.75, huber_epsilon=1e-4
- **N_teacher annealing:** 64->1280 over 100k iterations
- **Changes from exp_2:** student_steps 8->16 (segments 6.25% instead of 12.5%)
- **Result:** Similar to exp_2. 8 and 16 step students match each other but not teacher.
- **Checkpoints:** saved_state/checkpoint_{0,25,50,...}.pt

### darcy_student/exp_4/
- **Config:** student_steps=16, batch_size=64, grad_accum=4 (effective batch=256), lr=1e-4, epochs=1000
- **Teacher:** darcy_teacher/exp_1/saved_state/checkpoint_200.pt (raw weights)
- **CD params:** ema_rate=0.9999, x_var_frac=0.75, huber_epsilon=1e-4
- **N_teacher annealing:** 64->1280 over 100k iterations
- **Changes from exp_3:** gradient accumulation (4 steps) to simulate 4x larger batch size
- **Hypothesis:** Larger effective batch gives cleaner gradients, may reduce mode collapse
- **Result:** TBD
- **Checkpoints:** saved_state/checkpoint_{0,25,50,...}.pt

### darcy_student/exp_5/
- **Config:** student_steps=16, batch_size=64, grad_accum=8 (effective batch=512), lr=1e-4, epochs=1000
- **Teacher:** darcy_teacher/exp_1/saved_state/checkpoint_200.pt (raw weights)
- **CD params:** ema_rate=0.9999, x_var_frac=0.75, huber_epsilon=1e-4
- **N_teacher annealing:** 64->1280 over 100k iterations
- **Changes from exp_4:** gradient accumulation 4->8 (effective batch 256->512)
- **Hypothesis:** Even larger effective batch, closer to paper's 2048
- **Result:** TBD
- **Checkpoints:** saved_state/checkpoint_{0,25,50,...}.pt
