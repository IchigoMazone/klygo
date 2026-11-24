"""Classical dataset-mining tools for object-detection annotations.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/mining/Document_Mining_Overview.ipynb

Public API (8 functions, 5 report classes across 3 submodules):
    Analysis (`analysis.py`):
        1. MiningStats          Summary statistics container.
        2. MiningScores         Per-image scoring report.
        3. MiningComparison     Before/after dataset comparison report.
        4. stats(...)           Compute dataset-level annotation and image statistics.
        5. score(...)           Score images by density, rarity, or difficulty heuristics.
        6. compare(...)         Compare two dataset states or splits.

    Duplicates (`duplicates.py`):
        7. DuplicateReport      Near-duplicate image cluster report.
        8. find_duplicates(...) Identify exact or perceptual near-duplicate images.
        9. deduplicate(...)     Remove redundant near-duplicate samples.

    Curation (`curation.py`):
       10. ClusterResult        Feature clustering report with cluster representatives.
       11. cluster(...)         Group images into visual/annotation clusters.
       12. select(...)          Select representative or high-value samples.
       13. balance(...)         Rebalance class representation across samples.
"""

from .analysis import MiningComparison, MiningScores, MiningStats, compare, score, stats
from .curation import ClusterResult, balance, boxes, classes, cluster, select
from .duplicates import DuplicateReport, deduplicate, find_duplicates

__all__ = [
    "MiningStats",
    "MiningScores",
    "DuplicateReport",
    "ClusterResult",
    "MiningComparison",
    "stats",
    "score",
    "find_duplicates",
    "deduplicate",
    "cluster",
    "classes",
    "boxes",
    "select",
    "balance",
    "compare",
]
