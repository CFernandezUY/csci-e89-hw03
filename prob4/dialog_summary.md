# Problem 4 Dialog Summary

## Initial requirements

The assignment goal was to reproduce the “Building an Image Classifier with
PyTorch” section of chapter 10 of *Hands-On Machine Learning with
Scikit-Learn and PyTorch* (Geron, 2025): Fashion-MNIST, a 784-300-100-10 MLP,
SGD, and a training-accuracy plot.

Before writing code, the requested rules were:

- Put each requested script in `prob4/scripts/` under the exact requested name.
- Keep reusable code above an `if __name__ == "__main__":` demonstration so a
  script can run independently and be pasted into a notebook.
- Do not use `argparse`, `sys.argv`, or `matplotlib.use("Agg")`.
- Download data to `prob4/data/`.
- Ensure the root `.gitignore` contains `data/`, `datasets/`, `__pycache__/`,
  and `.ipynb_checkpoints/`.
- Use straightforward code and a short explanatory top-level docstring in
  every script.

I confirmed these requirements before making changes.

## Prompts and generated work, in order

1. **Setup script:** The request was for `step1_setup.py` to import PyTorch and
   NumPy, seed Python/NumPy/PyTorch with 42, select CUDA/MPS/CPU automatically,
   and print Python, NumPy, PyTorch, and device information. I added
   `set_seed`, `get_device`, and `print_environment`, then ran the script. It
   reported Python 3.14.4, NumPy 2.4.6, PyTorch 2.14.0+cu130, and CPU in this
   environment. I also updated `.gitignore` with the required entries.

2. **Data script:** The request was for `step2_data.py` to download the train
   and test Fashion-MNIST splits with torchvision, transform images using
   `v2.ToImage()` and `v2.ToDtype(torch.float32, scale=True)`, make a seeded
   55,000/5,000 train/validation split, create batch-size-32 loaders with
   training-only shuffling, and print data details. I added `get_transform`,
   `load_datasets`, `make_dataloaders`, and `print_data_summary`.

3. **Download clarification:** After the data script encountered an HTTP 403,
   the instruction was to keep the required `torchvision.datasets.FashionMNIST`
   code, `download=True`, and `root=prob4/data` exactly as written. I did not
   replace the data, synthesize examples, hard-code output, add mirrors, or add
   a download workaround. I acknowledged that later scripts might be unable to
   run to completion in this sandbox.

4. **Model script:** The request was for `step3_model.py` with an
   `ImageClassifier` module containing Flatten, Linear(784, 300), ReLU,
   Linear(300, 100), ReLU, and Linear(100, 10), returning logits without
   softmax, plus `build_model`. I added the module, `build_model`, and
   `count_parameters`. Its main block asserted an output shape of `(B, 10)` and
   266,610 trainable parameters. This script ran successfully.

5. **Training script:** The request was for `step4_train.py` with an explicit
   `nn.CrossEntropyLoss`/SGD (`lr=0.1`) loop for exactly ten epochs, batch size
   32, per-epoch training/validation loss and accuracy, and a final validation
   accuracy assertion above 80%. I added `evaluate` and `train_model`, including
   `optimizer.zero_grad()` for every training batch and evaluation under
   `model.eval()` and `torch.no_grad()`. The main block seeds the run, loads
   data, trains, and performs the requested final assertion.

6. **Plotting script:** The request was for `step5_plot.py` to plot training
   and validation accuracy and losses, save `prob4/scripts/training_curves.png`,
   and call `plt.show()`. I added `plot_history(history, output_path=...)` with
   two subplots. Its independent main block reuses Steps 1-4 and trains for
   three epochs before plotting when the Fashion-MNIST data is available.

7. **Test evaluation script:** The request was for `step6_evaluate.py` to use
   the test set only at the end, print final test accuracy, and print predicted
   and true class names plus top-three softmax probabilities for three test
   images. I added `evaluate_test_set`, which runs the test pass with
   `model.eval()` and `torch.no_grad()` after Step 4 training finishes.

8. **Save/load script:** The request was for `step7_save_load.py` to save a
   `state_dict` and constructor hyperparameters to
   `prob4/my_fashion_mnist_model.pt`, reload using
   `torch.load(..., weights_only=True)`, rebuild `ImageClassifier`, and assert
   identical outputs on a test batch. I updated `ImageClassifier` to retain its
   constructor hyperparameters and added `save_model`, `load_model`, and
   `assert_identical_outputs`. The saved checkpoint contains the weights and
   hyperparameter dictionary; the reloaded model is checked with `torch.equal`.

9. **Silent-bug review:** The request was to review all scripts for missing
   `optimizer.zero_grad()`, missing evaluation/no-gradient contexts, softmax
   before cross-entropy, and test-set leakage. I reviewed all seven scripts and
   found none of those defects. No code change was necessary for that review.

## Errors encountered and their resolution

The only runtime problem was a sandbox network restriction when torchvision
attempted the required Fashion-MNIST download. The upstream URL returned HTTP
403 for `train-images-idx3-ubyte.gz`. This occurred when running the scripts
whose independent main blocks load Fashion-MNIST: Steps 2, 4, 5, 6, and 7.

This was not treated as a code bug. Per the explicit instruction, the required
Fashion-MNIST download implementation was retained unchanged, and no fallback
dataset, synthetic data, alternate mirror, or hard-coded output was added.

The model-only smoke test in Step 3 succeeded. Static compilation checks for
the Python scripts also succeeded. The training, plotting, final test
evaluation, and checkpoint round-trip could not be completed here because all
depend on the unavailable download.

## Final results

- **Model architecture check:** Passed: a batch of 32 images produced output
  shape `torch.Size([32, 10])` and the model had 266,610 trainable parameters.
- **Final validation accuracy:** Not available in this sandbox. Training could
  not start because Fashion-MNIST could not be downloaded, so the requested
  `> 80%` assertion was not reached or claimed as passed.
- **Final test accuracy:** Not available in this sandbox. The test evaluation
  is correctly placed after training, but training could not start without the
  dataset.
- **Training plot:** Not generated in this sandbox because the brief training
  run could not start. The script is configured to save it as
  `prob4/scripts/training_curves.png` when run where Fashion-MNIST is
  available.
- **Saved checkpoint:** Not generated in this sandbox because the standalone
  checkpoint workflow trains before saving. The save/load implementation is in
  place for a local run with the dataset available.

All code changes made during the implementation steps were committed on the
current branch, and pull-request metadata was created after each change.
