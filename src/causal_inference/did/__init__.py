"""Difference-in-Differences (DiD) estimation module."""

from .did_estimator import did_2x2, check_parallel_trends
from .event_study import event_study, plot_event_study
from .staggered import (
    StaggeredData,
    create_staggered_data,
    identify_cohorts,
    twfe_staggered,
)
from .callaway_santanna import callaway_santanna_ate
from .sun_abraham import sun_abraham_ate
from .comparison import compare_did_methods, demonstrate_twfe_bias

__all__ = [
    "did_2x2",
    "check_parallel_trends",
    "event_study",
    "plot_event_study",
    "StaggeredData",
    "create_staggered_data",
    "identify_cohorts",
    "twfe_staggered",
    "callaway_santanna_ate",
    "sun_abraham_ate",
    "compare_did_methods",
    "demonstrate_twfe_bias",
]
