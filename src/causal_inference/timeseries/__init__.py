"""
Time Series Causal Inference Module.

Sessions 135-137: Granger causality, VAR, PCMCI, and Structural VAR.
Session 145: Added KPSS, Phillips-Perron, and Johansen cointegration tests.

This module provides:
- Granger causality tests (pairwise and multivariate)
- VAR (Vector Autoregression) estimation
- Stationarity tests (ADF)
- Lag selection via information criteria
- PCMCI algorithm for time-series causal discovery
- Structural VAR (SVAR) with Cholesky identification
- Impulse Response Functions (IRF) with bootstrap inference
- Forecast Error Variance Decomposition (FEVD)

Example: Granger Causality
--------------------------
>>> from causal_inference.timeseries import granger_causality
>>> import numpy as np
>>> np.random.seed(42)
>>> n = 200
>>> x = np.cumsum(np.random.randn(n))
>>> y = np.zeros(n)
>>> for t in range(1, n):
...     y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + np.random.randn()
>>> result = granger_causality(np.column_stack([y, x]), lags=2)
>>> print(f"Granger causes: {result.granger_causes}")

Example: PCMCI
--------------
>>> from causal_inference.timeseries import pcmci
>>> import numpy as np
>>> np.random.seed(42)
>>> n = 200
>>> data = np.random.randn(n, 3)  # 3 variables
>>> # Create causal structure: X0 -> X1 -> X2 (lagged)
>>> for t in range(1, n):
...     data[t, 1] = 0.6 * data[t-1, 0] + 0.3 * data[t-1, 1] + np.random.randn() * 0.5
...     data[t, 2] = 0.5 * data[t-1, 1] + 0.2 * data[t-1, 2] + np.random.randn() * 0.5
>>> result = pcmci(data, max_lag=2, alpha=0.05)
>>> print(f"Found {len(result.links)} causal links")

Example: Structural VAR
-----------------------
>>> from causal_inference.timeseries import var_estimate, cholesky_svar, compute_irf
>>> import numpy as np
>>> np.random.seed(42)
>>> data = np.random.randn(200, 3)
>>> var_result = var_estimate(data, lags=2)
>>> svar_result = cholesky_svar(var_result)
>>> irf = compute_irf(svar_result, horizons=20)
>>> print(f"IRF shape: {irf.irf.shape}")  # (3, 3, 21)
"""

from causal_inference.timeseries.types import (
    GrangerResult,
    MultiGrangerResult,
    VARResult,
    ADFResult,
    LagSelectionResult,
    KPSSResult,
    PPResult,
    JohansenResult,
    VECMResult,
)

from causal_inference.timeseries.granger import (
    granger_causality,
    granger_causality_matrix,
    bidirectional_granger,
)

from causal_inference.timeseries.var import (
    var_estimate,
    var_forecast,
    var_residuals,
)

from causal_inference.timeseries.stationarity import (
    adf_test,
    difference_series,
    check_stationarity,
    kpss_test,
    phillips_perron_test,
    confirmatory_stationarity_test,
)

from causal_inference.timeseries.cointegration import (
    johansen_test,
    engle_granger_test,
)

from causal_inference.timeseries.lag_selection import (
    select_lag_order,
    compute_aic,
    compute_bic,
    compute_hqc,
)

from causal_inference.timeseries.pcmci_types import (
    TimeSeriesLink,
    LinkType,
    LaggedDAG,
    PCMCIResult,
    ConditionSelectionResult,
    CITestResult,
)

from causal_inference.timeseries.ci_tests_timeseries import (
    parcorr_test,
    cmi_knn_test,
    run_ci_test,
    get_ci_test,
)

from causal_inference.timeseries.pcmci import (
    pcmci,
    pcmci_plus,
    pc_stable_condition_selection,
    mci_test_all,
    run_granger_style_pcmci,
)

from causal_inference.timeseries.svar_types import (
    SVARResult,
    IRFResult,
    FEVDResult,
    FEVDBootstrapResult,
    HistoricalDecompositionResult,
    IdentificationMethod,
)

from causal_inference.timeseries.svar import (
    cholesky_svar,
    short_run_svar,
    long_run_svar,
    companion_form,
    vma_coefficients,
    structural_vma_coefficients,
    check_stability,
    long_run_impact_matrix,
    verify_identification,
)

from causal_inference.timeseries.irf import (
    compute_irf,
    compute_irf_reduced_form,
    bootstrap_irf,
    irf_significance_test,
    asymptotic_irf_se,
    moving_block_bootstrap_irf,
    moving_block_bootstrap_irf_joint,
    joint_confidence_bands,
)

from causal_inference.timeseries.fevd import (
    compute_fevd,
    historical_decomposition,
    fevd_convergence,
    variance_contribution_table,
    bootstrap_fevd,
)

from causal_inference.timeseries.vecm import (
    vecm_estimate,
    vecm_forecast,
    vecm_granger_causality,
    compute_error_correction_term,
)

__all__ = [
    # Types
    "GrangerResult",
    "MultiGrangerResult",
    "VARResult",
    "ADFResult",
    "LagSelectionResult",
    "KPSSResult",
    "PPResult",
    "JohansenResult",
    "VECMResult",
    # Granger
    "granger_causality",
    "granger_causality_matrix",
    "bidirectional_granger",
    # VAR
    "var_estimate",
    "var_forecast",
    "var_residuals",
    # Stationarity
    "adf_test",
    "difference_series",
    "check_stationarity",
    "kpss_test",
    "phillips_perron_test",
    "confirmatory_stationarity_test",
    # Cointegration
    "johansen_test",
    "engle_granger_test",
    # Lag selection
    "select_lag_order",
    "compute_aic",
    "compute_bic",
    "compute_hqc",
    # PCMCI Types
    "TimeSeriesLink",
    "LinkType",
    "LaggedDAG",
    "PCMCIResult",
    "ConditionSelectionResult",
    "CITestResult",
    # PCMCI CI tests
    "parcorr_test",
    "cmi_knn_test",
    "run_ci_test",
    "get_ci_test",
    # PCMCI algorithm
    "pcmci",
    "pcmci_plus",
    "pc_stable_condition_selection",
    "mci_test_all",
    "run_granger_style_pcmci",
    # SVAR Types
    "SVARResult",
    "IRFResult",
    "FEVDResult",
    "FEVDBootstrapResult",
    "HistoricalDecompositionResult",
    "IdentificationMethod",
    # SVAR core
    "cholesky_svar",
    "short_run_svar",
    "long_run_svar",
    "companion_form",
    "vma_coefficients",
    "structural_vma_coefficients",
    "check_stability",
    "long_run_impact_matrix",
    "verify_identification",
    # IRF
    "compute_irf",
    "compute_irf_reduced_form",
    "bootstrap_irf",
    "irf_significance_test",
    "asymptotic_irf_se",
    "moving_block_bootstrap_irf",
    "moving_block_bootstrap_irf_joint",
    "joint_confidence_bands",
    # FEVD
    "compute_fevd",
    "historical_decomposition",
    "fevd_convergence",
    "variance_contribution_table",
    "bootstrap_fevd",
    # VECM
    "vecm_estimate",
    "vecm_forecast",
    "vecm_granger_causality",
    "compute_error_correction_term",
]
