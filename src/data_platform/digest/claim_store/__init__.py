"""Human-approved claims with verified retrieval and typed comparisons."""

from .comparison import COMPARISON_PROMPT as COMPARISON_PROMPT
from .models import (
    CheckReport,
    ClaimRecord,
    ComparisonDecision,
    PairComparison,
    RetrievalMatch,
    SyncResult,
)
from .models import (
    Status as Status,
)
from .models import (
    Verdict as Verdict,
)
from .store import ClaimStore

__all__ = [
    "ClaimStore",
    "ClaimRecord",
    "ComparisonDecision",
    "RetrievalMatch",
    "PairComparison",
    "CheckReport",
    "SyncResult",
]

# Preserve type reprs and pickle import paths after moving their definitions.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
