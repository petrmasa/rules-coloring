# Item Importance in Association Rules — Reproducibility Repository


## Repository content

*  *show_coloring.py* - example to run coloring of the rules
*  *show_pruning_phases.py* - example to run pruning and show results
*  *bench_*.py* - code to reproduce benchmarks
*  *run_all.py* - runs all tests
*  */lib* - the folder with the library
*  */algorithms* - supporting code for other pruning algorithms
*  */output* - outputs from the test run
*  */data* - pre-downloaded datasets



## Installation and running

1. Install requirements (pip install -r requirements.txt)
2. Run tests via python run_all.py (or with parameters)

Results are saved to `output/` folder.


Run a single experiment by name, filter to one dataset, skip experiments, or combine:

```bash
python run_all.py titanic                        # all experiments, titanic only
python run_all.py pruning_comparison             # one experiment, all datasets
python run_all.py titanic pruning_comparison     # one experiment, titanic only
python run_all.py --skip uci_comparison          # all except uci_comparison
python run_all.py titanic --skip uci_comparison  # titanic, skip uci_comparison
python run_all.py coloring_speed pruning_speed   # multiple experiments, all datasets
```

### Individual scripts

All scripts (except `bench_uci_comparison.py`) accept one or more dataset
names as arguments to restrict output to that subset:

```bash
python show_coloring.py titanic
python bench_inert_analysis.py titanic diabetes
python bench_pruning_comparison.py titanic
python bench_coloring_speed.py titanic
python bench_pruning_speed.py titanic
```
---

## Key library API

```python
from cleverminer import cleverminer
from lib.action_literals import action_literals
from lib.pruning_by_colors import prune_by_colors, derive_and_prune

# Color a single rule
al = action_literals(clm)
text, stats = al._get_literal_importance(rule_id=1, print_rule=False,
                                          print_details=False, get_stats=True)
# stats['has_ignorable_literal']  — True if any attribute is red
# stats['has_ignorable_category'] — True if any eligible value is red

# Phase 1 pruning
ignore_ids = prune_by_colors(clm, prune_yellow=False)

# Phase 2 pruning (mutates clm.result['rules'] — restore after use)
p1_ignore, p2_ignore, derived_ids = derive_and_prune(clm, prune_yellow=False)
```
