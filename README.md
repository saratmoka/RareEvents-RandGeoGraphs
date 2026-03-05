# Rare Events in Random Geometric Graphs

This repository provides Python code for estimating rare-event probabilities in random geometric graphs (Gilbert graphs) on a square window, using the following three methods:

- Naïve Monte Carlo;
- Conditional Monte Carlo;
- Importance Sampling based Monte Carlo.

Six rare events are covered:

| Module | Rare Event |
|--------|-----------|
| `ec.py` | Edge count does not exceed a threshold |
| `md.py` | Maximum degree does not exceed a threshold |
| `mcc.py` | Maximum connected component size does not exceed a threshold |
| `ntg.py` | Number of triangles does not exceed a threshold |
| `mcs.py` | Maximum clique size does not exceed a threshold |
| `planar.py` | Graph is non-planar |

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

# Reference
S. Moka, C. Hirsch, V. Schmidt, D. Kroese (2025). *Efficient Rare-Event Simulation for Random Geometric Graphs via Importance Sampling*. arXiv:2504.10530. https://arxiv.org/abs/2504.10530
