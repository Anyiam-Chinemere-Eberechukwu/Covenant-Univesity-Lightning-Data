# Covenant-Univesity-Lightning-Data
This is an extraction of lightning strikes that occurred from 2015 to 2023 within a 2km radius of the medical centre at Covenant University. 

## Peak-current probability analysis

`lightning_probability.py` builds the cumulative probability curve, the probability distribution and a probability matrix of lightning peak current (kA), with **positive and negative strokes treated separately** and a log-normal fit (as in IEC 62305 / CIGRE).

```bash
pip install pandas numpy scipy matplotlib
python lightning_probability.py                  # cloud-to-ground strokes (default)
python lightning_probability.py --type all       # ground + intra-cloud
python lightning_probability.py --type cloud     # intra-cloud only
python lightning_probability.py big.csv --chunksize 1000000 --label "Nigeria, 2015–2021"   # large files
```

Outputs go to `results/`: `cumulative_probability_*.png`, `probability_distribution_*.png`, `summary_*.csv` (median, σ, and currents exceeded with 95/90/80/50/20/10/5/1 % probability) and `probability_matrix_*.csv` (P(I > threshold) for 2–200 kA).

Cloud-to-ground results, 2015–2023 (midpoint = 50 % probability):

| Polarity | Strokes | Median (log-normal) | σ ln I | 95 % exceed | 5 % exceed |
|---|---|---|---|---|---|
| Negative | 1,190 | 20.8 kA | 0.61 | 7.6 kA | 56.8 kA |
| Positive | 593 | 17.2 kA | 0.56 | 6.9 kA | 43.2 kA |
