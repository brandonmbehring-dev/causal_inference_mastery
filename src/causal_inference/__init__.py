"""Causal inference methods: RCT, PSM, DiD, IV, RDD, Selection, Bounds, QTE, MTE, Mediation, CF, SS."""

# Shift-Share IV module (Session 94)
from .shift_share import (
    ShiftShareIV,
    shift_share_iv,
    ShiftShareResult,
    RotembergDiagnostics,
)

# Control Function module (Session 93)
from .control_function import (
    # Linear Control Function
    ControlFunction,
    control_function_ate,
    # Nonlinear Control Function (Probit/Logit)
    NonlinearControlFunction,
    nonlinear_control_function,
    # Types
    ControlFunctionResult,
    FirstStageResult,
    NonlinearCFResult,
)

# Mediation module (Session 92)
from .mediation import (
    # Baron-Kenny
    baron_kenny,
    # Natural effects
    mediation_analysis,
    natural_direct_effect,
    natural_indirect_effect,
    # Controlled direct effect
    controlled_direct_effect,
    # Sensitivity analysis
    mediation_sensitivity,
    # Types
    MediationResult,
    BaronKennyResult,
    CDEResult,
    SensitivityResult,
)

# MTE module (Session 90)
from .mte import (
    # LATE estimation
    late_estimator,
    late_bounds,
    complier_characteristics,
    # MTE estimation
    local_iv,
    polynomial_mte,
    # Policy parameters
    ate_from_mte,
    att_from_mte,
    atu_from_mte,
    prte,
    late_from_mte,
    # Diagnostics
    common_support_check,
    mte_sensitivity_to_trimming,
    monotonicity_test,
    propensity_variation_test,
    mte_shape_test,
    # Types
    MTEResult,
    LATEResult,
    PolicyResult,
    ComplierResult,
    CommonSupportResult,
)

# Bounds module (Sessions 86-87)
from .bounds import (
    # Manski bounds
    manski_worst_case,
    manski_mtr,
    manski_mts,
    manski_mtr_mts,
    manski_iv,
    compare_bounds,
    # Lee bounds
    lee_bounds,
    lee_bounds_tightened,
    check_monotonicity,
    # Types
    ManskiBoundsResult,
    ManskiIVBoundsResult,
    LeeBoundsResult,
)

# QTE module (Session 88)
from .qte import (
    # Unconditional QTE
    unconditional_qte,
    unconditional_qte_band,
    # Conditional QTE
    conditional_qte,
    conditional_qte_band,
    # RIF-OLS
    rif_qte,
    rif_qte_band,
    # Types
    QTEResult,
    QTEBandResult,
)
