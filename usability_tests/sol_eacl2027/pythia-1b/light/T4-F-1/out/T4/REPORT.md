# Participant self-report

- Final artifact path: `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The checkpoints are about 2 GB apiece, so the script keeps the input files memory-mapped and limits float32 intermediates to one MLP tensor at a time.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): PyTorch 2.14.0 for equality checks and float32 task-vector arithmetic; safetensors 0.5.3 for memory-mapped checkpoint reads and the single-file output, chosen for direct and lossless checkpoint handling.
- Approximate time spent, if you can tell: About 5 minutes.
