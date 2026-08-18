# Post-deployment performance report

- Endpoint: `http://localhost:8000/predict`
- Samples: 30 (15 requested per class)
- **Live accuracy: 100.00%** (30/30)

## Confusion matrix (rows = true, columns = predicted)

| true \ predicted | cat | dog |
| --- | --- | --- |
| cat | 15 | 0 |
| dog | 0 | 15 |
