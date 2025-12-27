"""R Triangulation tests for external validation against R packages.

This module provides Layer 5 validation by comparing our Python implementations
against established R packages:
- PStrata (Li & Li 2023) for Principal Stratification
- DTRreg (Wallace et al. 2017) for Dynamic Treatment Regimes
- did (Callaway & Sant'Anna 2021) for Difference-in-Differences
- rdrobust (Calonico, Cattaneo & Titiunik 2014) for RDD
- rddensity (Cattaneo, Jansson & Ma 2020) for McCrary density tests
- AER (Kleiber & Zeileis 2008) for Instrumental Variables
- Synth (Abadie et al.) for Synthetic Control Method
- grf (Athey, Tibshirani & Wager 2019) for Causal Forests

Tests skip gracefully when R/rpy2 is unavailable.
"""

from .r_interface import (
    # Availability checks
    check_r_available,
    check_pstrata_installed,
    check_dtrreg_installed,
    check_did_installed,
    check_rdrobust_installed,
    check_rddensity_installed,
    check_aer_installed,
    check_synth_installed,
    check_grf_installed,
    # DiD
    r_did_callaway_santanna,
    # RDD
    r_rdd_rdrobust,
    r_rdd_mccrary,
    # IV
    r_2sls_aer,
    r_liml_aer,
    # SCM
    r_scm_synth,
    # CATE (Session 125)
    r_causal_forest_grf,
    # DTR (Session 125)
    r_q_learning_dtrreg,
    r_a_learning_dtrreg,
)

__all__ = [
    # Availability checks
    "check_r_available",
    "check_pstrata_installed",
    "check_dtrreg_installed",
    "check_did_installed",
    "check_rdrobust_installed",
    "check_rddensity_installed",
    "check_aer_installed",
    "check_synth_installed",
    "check_grf_installed",
    # DiD
    "r_did_callaway_santanna",
    # RDD
    "r_rdd_rdrobust",
    "r_rdd_mccrary",
    # IV
    "r_2sls_aer",
    "r_liml_aer",
    # SCM
    "r_scm_synth",
    # CATE (Session 125)
    "r_causal_forest_grf",
    # DTR (Session 125)
    "r_q_learning_dtrreg",
    "r_a_learning_dtrreg",
]
