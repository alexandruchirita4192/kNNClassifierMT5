# kNN MT5 Strategy

## Algorithm
k-Nearest Neighbors classifier

- Distance-based model
- Non-parametric
- Sensitive to feature scaling

## Disclaimer: It has not produced any transactions on a long period of time (6 months) on XAGUSD so it is not used for trading, it is only as proof of concept that it runs (not doing anything, but runs).

## Training

```text
python train_mt5_knn_classifier.py --symbol XAGUSD --timeframe M15 --bars 80000 --horizon-bars 8 --train-ratio 0.82 --output-dir output_knn_XAGUSD_M15_h8_82
```

## Outputs

- ml_strategy_classifier_knn.onnx
- model_metadata.json

## MT5 Usage

1. Copy ONNX to MQL5/Files
2. Compile EA (model is embedded)
3. Run Strategy Tester

## Notes

- Faster than SVM
- Produces more trades
- More sensitive to noise
- Good for short-term patterns

## Tuning

- n_neighbors: 10–50
- threshold parameters from metadata
