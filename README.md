# Rare Events in Random Geometric Graphs

This repository provides Python code for estimating rare-event probabilities in random geometric graphs (Gilbert graphs) on a square window. Six rare events are covered:

| Module | Rare Event |
|--------|-----------|
| `ec.py` | Edge count does not exceed a threshold |
| `md.py` | Maximum degree does not exceed a threshold |
| `mcc.py` | Maximum connected component size does not exceed a threshold |
| `ntg.py` | Number of triangles does not exceed a threshold |
| `mcs.py` | Maximum clique size does not exceed a threshold |
| `planar.py` | Graph is non-planar |

# Algorithms

Three estimators are implemented for each example.

**Naïve Monte Carlo (NMC):** Generate independent realisations of the Gilbert graph and compute the fraction that satisfy the rare event. Simple but highly inefficient for small probabilities.

**Conditional Monte Carlo (CMC):** Points are added sequentially to the graph one at a time. At each step the contribution to the probability estimate is computed analytically by conditioning on the current point configuration. This yields substantially lower variance than NMC. See [Hirsch, Moka, Taimre & Kroese (2022)](https://link.springer.com/article/10.1007/s11009-021-09857-7) for details.

**Importance Sampling (IS):** Points are again added sequentially, but cells of the window that would definitely violate the rare event are *blocked* from receiving new points. The resulting change of measure is corrected by a likelihood ratio. This concentrates sampling effort on configurations consistent with the rare event, giving substantially lower variance than CMC. See [Moka, Hirsch, Schmidt & Kroese (2025)](https://arxiv.org/abs/2504.10530) for details.

# Performance

The table below compares CMC and IS at a precision target of RV/m < 0.01, with window W = [0,10]², r = 1, and a 100×100 IS grid. Probabilities are approximately 10⁻⁴. CMC times marked † are extrapolated from a pilot run.

| Example | Z | RV (CMC) | Time CMC (s) | RV (IS) | Time IS (s) | Speedup |
|---------|---|----------|--------------|---------|-------------|---------|
| EC      | 1.15×10⁻⁴ | 17.6 | 1.4   | 9.11 | 0.02 | ~70×   |
| MD      | 9.38×10⁻⁴ | 51.4 | 0.7   | 6.82 | 0.02 | ~35×   |
| MCC     | 2.48×10⁻⁴ | 211  | 2.4   | 19.9 | 0.02 | ~120×  |
| NTG     | 1.91×10⁻⁴ | 189  | 2.0   | 5.65 | 0.01 | ~200×  |
| MCS     | 1.91×10⁻⁴ | 189  | 2.0   | 5.65 | 0.01 | ~200×  |
| Planarity | 1.42×10⁻⁴ | 192 | 498† | 11.4 | 66.0 | ~8×  |

IS achieves 35–200× wall-clock speedup over CMC for the threshold examples. For Planarity, CMC is infeasible at this probability level while IS converges in about a minute.

# Dependencies
```
Python      (>= 3.12)
NumPy       (>= 2.2.4)
SciPy       (>= 1.14.1)
Numba       (>= 0.61.0)
NetworkX    (>= 3.4.2)
IPython     (>= 8.27.0)
Jupyter-Notebook (>= 7.2.2)
```

# Download Instructions
Download the following files from this repository to a local folder on your computer. Note that all files must be saved in the same folder.
```
Rare-Event-Simulation.ipynb
ec.py          ec_workspace.py
md.py          md_workspace.py
mcc.py         mcc_workspace.py
ntg.py         ntg_workspace.py
mcs.py         mcs_workspace.py
planar.py      planar_workspace.py
image.png
```

# Running Instructions
Open the Jupyter notebook *Rare-Event-Simulation.ipynb*. The notebook contains instructions and examples for estimating rare-event probabilities using all three methods. Each `*_workspace.py` file provides a ready-to-run script for the corresponding example.

# References
- C. Hirsch, S. B. Moka, T. Taimre & D. P. Kroese (2022). *Rare Events in Random Geometric Graphs*. Methodology and Computing in Applied Probability, 24, 1367–1383. https://link.springer.com/article/10.1007/s11009-021-09857-7

- S. Moka, C. Hirsch, V. Schmidt & D. P. Kroese (2025). *Efficient Rare-Event Simulation for Random Geometric Graphs via Importance Sampling*. arXiv:2504.10530. https://arxiv.org/abs/2504.10530
