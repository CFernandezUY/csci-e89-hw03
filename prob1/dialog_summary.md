# Dialog Summary — Fashion-MNIST MLP (Chapter 10 Reproduction)

Assignment: reproduce "Building an Image Classifier with PyTorch" (Chapter 10,
*Hands-On Machine Learning with Scikit-Learn and PyTorch*, Geron, 2025) —
Fashion-MNIST, a 784-300-100-10 MLP, plain SGD, and a training-accuracy plot.
Built step by step; each script lives in `prob1/scripts/`, keeps reusable
code above an `if __name__ == "__main__":` demo block, runs standalone, and
was committed once verified working.

## 1. Kickoff prompt

**Prompt:** Described the assignment goal and the workflow rules (one script
per step in `prob1/scripts/`, runnable standalone, reusable code above
`if __name__ == "__main__":`, commit each script once it works, don't start
until instructed).

**Generated:** Nothing yet — confirmed understanding of the rules and
architecture, waited for step 1.

**Errors:** None.

## 2. step1_setup.py

**Prompt:** Import torch/numpy, fix the seed to 42 for random/numpy/torch,
pick CUDA/MPS/CPU automatically, print Python/NumPy/PyTorch versions and the
device.

**Generated:** `set_seed()` (seeds `random`, `numpy`, `torch`, `torch.cuda`)
and `get_device()` (CUDA → MPS → CPU fallback), with a demo block printing
versions, seed, and device.

**Errors hit and fixed:**
- Neither `torch` nor `numpy` were installed in the container. Fixed by
  `pip3 install --user numpy torch`.
- First attempt used the PyTorch CPU-only wheel index
  (`download.pytorch.org/whl/cpu`), which the environment's proxy rejected
  with `403 Forbidden`. Fixed by installing from the default PyPI index
  instead, which succeeded (pulling the CUDA-enabled build, which still runs
  fine and correctly falls back to CPU since no GPU is present).

**Result:** Ran cleanly — Python 3.11.15, NumPy 2.4.6, PyTorch 2.14.0+cu130,
device `cpu`. Committed.

## 3. step2_data.py

**Prompt:** Load FashionMNIST via torchvision (train and test), convert
images to float32 tensors in [0,1] with `transforms.v2` (`ToImage` +
`ToDtype(scale=True)`), split the 60,000 training images into 55,000
train / 5,000 validation with a seeded generator, and build DataLoaders with
batch size 32 (shuffle only train). Print split sizes, one batch's shapes,
and the class names.

**Generated:** `load_datasets()`, `split_train_val()` (seeded
`torch.Generator` + `random_split`), and `build_dataloaders()`, plus a demo
block printing dataset sizes, one batch's tensor shapes/dtypes, the pixel
value range, and the 10 Fashion-MNIST class names.

**Errors hit and fixed:**
- `torchvision` wasn't installed. Fixed with `pip3 install --user
  torchvision` (pulled in Pillow as a dependency).
- Added a `.gitignore` for `prob1/data/` (and `__pycache__/`, `*.pyc`) so the
  downloaded dataset files aren't committed.

**Result:** Train/val/test sizes 55000/5000/10000; batch shape
`[32, 1, 28, 28]` float32 in `[0.000, 1.000]`; labels `[32]` int64; class
names printed correctly. Committed.

## 4. step3_model.py

**Prompt:** An `nn.Module` called `ImageClassifier` = Flatten,
Linear(784,300), ReLU, Linear(300,100), ReLU, Linear(100,10) returning raw
logits, plus a `build_model` function. Assert the output shape `(B,10)` and
the total parameter count 266,610.

**Generated:** `ImageClassifier` (flatten → hidden1+ReLU → hidden2+ReLU →
linear output, no softmax), `build_model()`, and `count_parameters()`, with
a demo block asserting a `(32, 10)` output shape and the 266,610 parameter
count on a dummy batch.

**Errors hit and fixed:** None — worked on the first run.

**Result:** Output shape `(32, 10)`, 266,610 total parameters, both
assertions passed. Committed.

## 5. step4_train.py

**Prompt:** A manual training loop using `CrossEntropyLoss` and SGD with
`lr=0.1` for exactly 10 epochs. Record train and validation loss AND
accuracy per epoch and print one line per epoch. Call
`optimizer.zero_grad()` every step and use `model.eval()` with
`torch.no_grad()` for validation.

**Generated:** `train_one_epoch()` (zeros gradients every batch, trains in
`model.train()` mode), `evaluate()` (`@torch.no_grad()` + `model.eval()`),
and `train_model()` tying them together for 10 epochs, returning a history
dict of per-epoch train/val loss and accuracy, with a demo block that trains
on Fashion-MNIST and prints one line per epoch.

**Errors hit and fixed:** None — worked on the first run (took ~78s on
CPU).

**Result:** Epoch 10/10 — train_loss 0.2525, train_acc 0.9042, val_loss
0.3333, val_acc 0.8766. Committed.

## 6. step5_plot.py

**Prompt:** Plot training accuracy and validation accuracy per epoch (plus a
second subplot with the losses) and save it as `training_curves.png`.

**Generated:** `plot_training_curves()` — a two-subplot figure (accuracy
left, loss right, train+val curves on each) saved to a PNG, plus a demo
block that retrains the model (seed 42, same as step4) and saves the plot.

**Errors hit and fixed:**
- `matplotlib` wasn't installed. Fixed with `pip3 install --user
  matplotlib`, using the non-interactive `Agg` backend since the container
  is headless.

**Result:** Reproduced the identical deterministic training run from step4
and saved `prob1/scripts/training_curves.png` (verified visually — clean
accuracy/loss curves matching the printed epoch values). Committed,
including the PNG as the requested deliverable.

## 7. step6_evaluate.py

**Prompt:** Report accuracy on the test set (used once, at the end) and,
for 3 test images, the predicted class, true class, and top-3 softmax
probabilities.

**Generated:** `test_accuracy()` (reuses step4's `evaluate()`, called only
after training) and `predict_samples()` (`@torch.no_grad()` +
`model.eval()`, applies `softmax` only for reporting, never before the
loss), with a demo block that trains, evaluates the test set once, and
prints predictions for the first 3 test images.

**Errors hit and fixed:** None — worked on the first run.

**Result:** Test accuracy 0.8715; all 3 sample predictions correct (Ankle
boot, Pullover, Trouser) with sharply peaked top-3 softmax distributions.
Committed.

## 8. step7_save_load.py

**Prompt:** Save the `state_dict` plus the constructor hyperparameters,
reload with `weights_only=True`, and assert the reloaded model gives
identical outputs.

**Generated:** `save_checkpoint()` (bundles `state_dict` with an explicit
hyperparameters dict describing the fixed architecture), `load_checkpoint()`
(`torch.load(..., weights_only=True)` then `load_state_dict`), with a demo
block that builds a model, records its output on a dummy batch, saves,
reloads, and asserts (`torch.equal`) the outputs are bit-identical plus the
hyperparameters round-trip unchanged.

**Errors hit and fixed:** None — worked on the first run. Added `*.pt` to
`.gitignore` so the ~1 MB checkpoint file (a generated artifact, unlike the
requested plot) isn't committed.

**Result:** Reload succeeded under `weights_only=True`; reloaded output was
bit-identical to the original; both assertions passed. Committed.

## 9. Bug review

**Prompt:** Review all scripts for four silent-bug patterns: missing
`zero_grad`, missing `model.eval()`/`no_grad` in validation, softmax applied
before `CrossEntropyLoss`, and any test-set leakage. Report findings and fix
anything wrong.

**Generated:** No code changes — a line-by-line audit of all 7 scripts
against the four patterns.

**Findings:** None of the four bug patterns were present:
- `zero_grad()` is called every training step (`step4_train.py:31`).
- Validation and test inference consistently use `@torch.no_grad()` +
  `model.eval()` (`step4_train.py`, `step6_evaluate.py`,
  `step7_save_load.py`), and `model.train()` is re-armed at the start of
  every training epoch.
- `CrossEntropyLoss` always receives raw logits; the only `softmax` call
  (`step6_evaluate.py:37`) is for displaying probabilities, not for loss.
- The test set is split off before any train/val split (`step2_data.py`),
  is unused during training/plotting (`step4_train.py`, `step5_plot.py`),
  and is touched exactly once, after training, in `step6_evaluate.py`.

**Result:** Clean bill of health; no changes or commits needed.

## Final state

All seven scripts committed to branch `claude/clever-hamilton-peodds` in
`prob1/scripts/`:

| Script | Purpose |
| --- | --- |
| `step1_setup.py` | Seeds, device selection, version info |
| `step2_data.py` | Fashion-MNIST loading, 55k/5k split, DataLoaders |
| `step3_model.py` | 784-300-100-10 `ImageClassifier` MLP |
| `step4_train.py` | Manual SGD training loop, 10 epochs |
| `step5_plot.py` | Accuracy/loss curves → `training_curves.png` |
| `step6_evaluate.py` | Final test accuracy + sample predictions |
| `step7_save_load.py` | Checkpoint save/load with `weights_only=True` |

Final metrics (seed 42, CPU, 10 epochs, SGD lr=0.1): train accuracy 90.4%,
validation accuracy 87.7% (epoch 10), test accuracy 87.2%.
