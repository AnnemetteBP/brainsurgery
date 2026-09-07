## Structural losslessness
The primary losslessness evaluation applied a value-preserving namespace move and restoration, without arithmetic or dtype conversion, to 10 checkpoints at native dtype and evaluated 700 prompt--model pairs. Exact tensor, behavioral, and non-finite checks are reported in Table~\ref{tab:behavioral-methodology}.

## Pythia precision ablation
The secondary dtype/precision ablation evaluated 8 Pythia model/dtype arms and 560 prompt--model pairs under the multiply-by-0.5 then multiply-by-2 save round trip. FP32 storage and inference were retained throughout the FP32 arm and were never cast back to FP16. The old native-FP16 arithmetic round trip is not the primary losslessness test. All negative results are retained.
