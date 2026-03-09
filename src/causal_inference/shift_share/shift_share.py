"""
Shift-Share (Bartik) Instrumental Variables Estimation.

The shift-share instrument exploits:
1. Cross-sectional variation in exposure to sectors (shares)
2. Time-series variation in aggregate shocks (shifts)

Instrument: Z_i = sum_s(share_{i,s} * shock_s)

Identification requires either:
- Exogeneity of shares (Goldsmith-Pinkham et al. 2020)
- Exogeneity of shocks (Borusyak et al. 2022)

References
----------
- Bartik (1991). Who Benefits from State and Local Economic Development Policies?
- Goldsmith-Pinkham, Sorkin, Swift (2020). Bartik Instruments
- Borusyak, Hull, Jaravel (2022). Quasi-Experimental Shift-Share Designs
"""

from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import statsmodels.api as sm

from .types import FirstStageResult, RotembergDiagnostics, ShiftShareResult


class ShiftShareIV:
    """
    Shift-Share (Bartik) Instrumental Variables estimator.

    Constructs an instrument from sector shares and aggregate shocks,
    then runs 2SLS with the constructed instrument.

    Parameters
    ----------
    inference : {'robust', 'clustered'}, default='robust'
        Standard error type. Use 'clustered' for panel data.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Attributes
    ----------
    result_ : ShiftShareResult
        Full estimation results after calling fit().
    instrument_ : NDArray[np.float64]
        Constructed Bartik instrument.
    fitted_ : bool
        Whether model has been fitted.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_inference.shift_share import ShiftShareIV
    >>>
    >>> # Regional labor market example
    >>> n_regions, n_sectors = 100, 10
    >>> shares = np.random.dirichlet(np.ones(n_sectors), n_regions)
    >>> shocks = np.random.normal(0.02, 0.05, n_sectors)
    >>> Z_bartik = shares @ shocks
    >>>
    >>> # Treatment and outcome
    >>> D = 2.0 * Z_bartik + np.random.normal(0, 0.5, n_regions)
    >>> Y = 1.5 * D + np.random.normal(0, 1, n_regions)
    >>>
    >>> # Estimate
    >>> ssiv = ShiftShareIV()
    >>> result = ssiv.fit(Y, D, shares, shocks)
    >>> print(f"Effect: {result['estimate']:.3f}")
    """

    def __init__(
        self,
        inference: Literal["robust", "clustered"] = "robust",
        alpha: float = 0.05,
    ) -> None:
        self.inference = inference
        self.alpha = alpha
        self.fitted_ = False
        self.result_: Optional[ShiftShareResult] = None
        self.instrument_: Optional[NDArray[np.float64]] = None

    def fit(
        self,
        Y: NDArray[np.floating],
        D: NDArray[np.floating],
        shares: NDArray[np.floating],
        shocks: NDArray[np.floating],
        X: Optional[NDArray[np.floating]] = None,
        clusters: Optional[NDArray] = None,
    ) -> ShiftShareResult:
        """
        Fit Shift-Share IV model.

        Parameters
        ----------
        Y : array-like, shape (n,)
            Outcome variable.
        D : array-like, shape (n,)
            Endogenous treatment variable.
        shares : array-like, shape (n, S)
            Sector shares for each observation. Rows should sum to ~1.
        shocks : array-like, shape (S,)
            Aggregate shocks to each sector.
        X : array-like, shape (n, k), optional
            Exogenous control variables.
        clusters : array-like, shape (n,), optional
            Cluster identifiers for clustered SEs.

        Returns
        -------
        ShiftShareResult
            TypedDict with estimation results and diagnostics.

        Raises
        ------
        ValueError
            If inputs are invalid or dimensions don't match.
        """
        # Validate and preprocess
        Y, D, shares, shocks, X, clusters = self._validate_inputs(Y, D, shares, shocks, X, clusters)
        n = len(Y)
        n_sectors = len(shocks)

        # Construct Bartik instrument
        Z_bartik = shares @ shocks
        self.instrument_ = Z_bartik

        # Check share normalization
        share_sums = shares.sum(axis=1)
        share_sum_mean = float(np.mean(share_sums))

        # Compute Rotemberg weights
        rotemberg = self._compute_rotemberg_weights(D, shares, shocks, X)

        # First stage: D ~ Z_bartik + X
        first_stage = self._first_stage(D, Z_bartik, X)

        # Second stage: 2SLS
        result = self._two_stage_ls(Y, D, Z_bartik, X, clusters)

        # Compile results
        self.result_ = ShiftShareResult(
            estimate=result["estimate"],
            se=result["se"],
            t_stat=result["t_stat"],
            p_value=result["p_value"],
            ci_lower=result["ci_lower"],
            ci_upper=result["ci_upper"],
            first_stage=first_stage,
            rotemberg=rotemberg,
            n_obs=n,
            n_sectors=n_sectors,
            share_sum_mean=share_sum_mean,
            inference=self.inference,
            alpha=self.alpha,
            message=self._generate_message(first_stage, rotemberg, share_sum_mean),
        )
        self.fitted_ = True
        return self.result_

    def _validate_inputs(
        self,
        Y: NDArray[np.floating],
        D: NDArray[np.floating],
        shares: NDArray[np.floating],
        shocks: NDArray[np.floating],
        X: Optional[NDArray[np.floating]],
        clusters: Optional[NDArray],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        Optional[NDArray[np.float64]],
        Optional[NDArray],
    ]:
        """Validate and preprocess inputs."""
        Y = np.asarray(Y, dtype=np.float64).ravel()
        D = np.asarray(D, dtype=np.float64).ravel()
        shares = np.asarray(shares, dtype=np.float64)
        shocks = np.asarray(shocks, dtype=np.float64).ravel()

        n = len(Y)

        # Check dimensions
        if len(D) != n:
            raise ValueError(f"Length mismatch: Y ({n}) != D ({len(D)})")
        if shares.ndim != 2:
            raise ValueError(f"shares must be 2D, got {shares.ndim}D")
        if shares.shape[0] != n:
            raise ValueError(f"shares rows ({shares.shape[0]}) != observations ({n})")
        if shares.shape[1] != len(shocks):
            raise ValueError(f"shares columns ({shares.shape[1]}) != shocks ({len(shocks)})")

        # Check for NaN/Inf
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            raise ValueError("NaN or infinite values in Y")
        if np.any(np.isnan(D)) or np.any(np.isinf(D)):
            raise ValueError("NaN or infinite values in D")
        if np.any(np.isnan(shares)) or np.any(np.isinf(shares)):
            raise ValueError("NaN or infinite values in shares")
        if np.any(np.isnan(shocks)) or np.any(np.isinf(shocks)):
            raise ValueError("NaN or infinite values in shocks")

        # Check minimum sample size
        if n < 10:
            raise ValueError(f"Insufficient sample size (n={n}). Need at least 10.")

        # Check for variation
        if np.std(D) < 1e-10:
            raise ValueError("No variation in treatment D")

        # Process X
        if X is not None:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if len(X) != n:
                raise ValueError(f"Length mismatch: Y ({n}) != X ({len(X)})")
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("NaN or infinite values in X")

        # Process clusters
        if clusters is not None:
            clusters = np.asarray(clusters)
            if len(clusters) != n:
                raise ValueError(f"Length mismatch: Y ({n}) != clusters ({len(clusters)})")

        # Warn if shares don't sum to ~1
        share_sums = shares.sum(axis=1)
        if np.abs(share_sums - 1.0).max() > 0.1:
            # Just a warning, not an error
            pass

        return Y, D, shares, shocks, X, clusters

    def _compute_rotemberg_weights(
        self,
        D: NDArray[np.float64],
        shares: NDArray[np.float64],
        shocks: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
    ) -> RotembergDiagnostics:
        """
        Compute Rotemberg (1983) weights for shift-share diagnostics.

        The weight for sector s is proportional to:
        alpha_s = shock_s * sum_i(share_{i,s} * D_i) / sum_s(...)

        These weights show which sectors drive the IV estimate.
        Negative weights indicate potential monotonicity violations.
        """
        n = len(D)
        n_sectors = len(shocks)

        # Residualize D on X if controls present
        if X is not None:
            X_with_const = np.column_stack([np.ones(n), X])
            D_resid = D - X_with_const @ np.linalg.lstsq(X_with_const, D, rcond=None)[0]
        else:
            D_resid = D - np.mean(D)

        # Compute raw weights: shock_s * sum_i(share_{i,s} * D_resid_i)
        raw_weights = np.zeros(n_sectors)
        for s in range(n_sectors):
            raw_weights[s] = shocks[s] * np.sum(shares[:, s] * D_resid)

        # Normalize to sum to 1
        total = np.sum(np.abs(raw_weights))
        if total > 1e-10:
            weights = raw_weights / total
        else:
            weights = np.ones(n_sectors) / n_sectors

        # Negative weight share
        negative_mask = weights < 0
        negative_weight_share = float(np.sum(np.abs(weights[negative_mask])))

        # Top 5 sectors by absolute weight
        abs_weights = np.abs(weights)
        top_5_idx = np.argsort(abs_weights)[-5:][::-1]
        top_5_weights = weights[top_5_idx]

        # Herfindahl index (concentration)
        herfindahl = float(np.sum(weights**2))

        return RotembergDiagnostics(
            weights=weights,
            negative_weight_share=negative_weight_share,
            top_5_sectors=top_5_idx,
            top_5_weights=top_5_weights,
            herfindahl=herfindahl,
        )

    def _first_stage(
        self,
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
    ) -> FirstStageResult:
        """Run first-stage regression: D ~ Z + X."""
        n = len(D)

        # Build design matrix
        if X is not None:
            design = np.column_stack([np.ones(n), Z, X])
        else:
            design = np.column_stack([np.ones(n), Z])

        # Fit OLS
        model = sm.OLS(D, design)
        fit = model.fit(cov_type="HC3")

        # Extract instrument coefficient (index 1)
        coef = fit.params[1]
        se = fit.bse[1]
        t_stat = fit.tvalues[1]
        f_stat = t_stat**2  # F = t^2 for single instrument
        f_pvalue = fit.pvalues[1]

        # Partial R-squared
        if X is not None:
            # Residualize Z on X
            X_with_const = np.column_stack([np.ones(n), X])
            Z_resid = Z - X_with_const @ np.linalg.lstsq(X_with_const, Z, rcond=None)[0]
            D_resid = D - X_with_const @ np.linalg.lstsq(X_with_const, D, rcond=None)[0]
            partial_r2 = float(np.corrcoef(Z_resid, D_resid)[0, 1] ** 2)
        else:
            partial_r2 = fit.rsquared

        weak_iv_warning = f_stat < 10

        return FirstStageResult(
            f_statistic=f_stat,
            f_pvalue=f_pvalue,
            partial_r2=partial_r2,
            coefficient=coef,
            se=se,
            t_stat=t_stat,
            weak_iv_warning=weak_iv_warning,
        )

    def _two_stage_ls(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        clusters: Optional[NDArray],
    ) -> dict:
        """Run 2SLS estimation."""
        n = len(Y)

        # Build design matrices
        if X is not None:
            exog_first = np.column_stack([np.ones(n), Z, X])
            exog_second = np.column_stack([np.ones(n), X])
            k_controls = X.shape[1]
        else:
            exog_first = np.column_stack([np.ones(n), Z])
            exog_second = np.ones((n, 1))
            k_controls = 0

        # First stage: get fitted D
        fs_model = sm.OLS(D, exog_first)
        fs_fit = fs_model.fit()
        D_hat = fs_fit.fittedvalues

        # Second stage: Y ~ D_hat + X
        if X is not None:
            ss_design = np.column_stack([np.ones(n), D_hat, X])
        else:
            ss_design = np.column_stack([np.ones(n), D_hat])

        ss_model = sm.OLS(Y, ss_design)

        # Fit with appropriate SEs
        if self.inference == "clustered" and clusters is not None:
            # Need to use original D for correct SEs
            if X is not None:
                final_design = np.column_stack([np.ones(n), D, X])
            else:
                final_design = np.column_stack([np.ones(n), D])

            # Manual 2SLS with clustered SEs via IV2SLS
            from linearmodels.iv import IV2SLS

            if X is not None:
                iv_result = IV2SLS(Y, exog_second, D.reshape(-1, 1), Z.reshape(-1, 1)).fit(
                    cov_type="clustered", clusters=clusters
                )
            else:
                iv_result = IV2SLS(Y, np.ones((n, 1)), D.reshape(-1, 1), Z.reshape(-1, 1)).fit(
                    cov_type="clustered", clusters=clusters
                )

            estimate = float(iv_result.params.iloc[0])
            se = float(iv_result.std_errors.iloc[0])
        else:
            # Use robust SEs with manual 2SLS correction
            ss_fit = ss_model.fit()

            # Get coefficient on D_hat
            estimate = ss_fit.params[1]

            # Correct SEs for 2SLS (use residuals from original D)
            if X is not None:
                final_design = np.column_stack([np.ones(n), D, X])
            else:
                final_design = np.column_stack([np.ones(n), D])

            resid = Y - final_design @ ss_fit.params

            # Robust SE computation
            bread = np.linalg.inv(ss_design.T @ ss_design)
            meat = ss_design.T @ np.diag(resid**2) @ ss_design
            vcov = n / (n - ss_design.shape[1]) * bread @ meat @ bread
            se = float(np.sqrt(vcov[1, 1]))

        # Inference
        t_stat = estimate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2 - k_controls))
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = estimate - z_crit * se
        ci_upper = estimate + z_crit * se

        return {
            "estimate": float(estimate),
            "se": float(se),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    def _generate_message(
        self,
        first_stage: FirstStageResult,
        rotemberg: RotembergDiagnostics,
        share_sum_mean: float,
    ) -> str:
        """Generate diagnostic message."""
        parts = []

        if first_stage["weak_iv_warning"]:
            parts.append(f"WARNING: Weak instrument (F={first_stage['f_statistic']:.1f} < 10)")

        if rotemberg["negative_weight_share"] > 0.1:
            parts.append(
                f"WARNING: {rotemberg['negative_weight_share']:.0%} negative Rotemberg weights"
            )

        if abs(share_sum_mean - 1.0) > 0.1:
            parts.append(f"NOTE: Share sums average {share_sum_mean:.2f} (expected ~1.0)")

        if not parts:
            parts.append("No diagnostic warnings")

        return "; ".join(parts)

    def summary(self) -> str:
        """Return formatted estimation summary."""
        if not self.fitted_:
            return "Model not fitted. Call fit() first."

        r = self.result_
        lines = [
            "=" * 65,
            "Shift-Share (Bartik) IV Estimation Results",
            "=" * 65,
            f"N observations:        {r['n_obs']}",
            f"N sectors:             {r['n_sectors']}",
            f"Inference:             {r['inference']}",
            "",
            "Treatment Effect:",
            f"  Estimate:            {r['estimate']:.4f}",
            f"  Std. Error:          {r['se']:.4f}",
            f"  t-statistic:         {r['t_stat']:.3f}",
            f"  p-value:             {r['p_value']:.4f}",
            f"  95% CI:              [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]",
            "",
            "First Stage:",
            f"  F-statistic:         {r['first_stage']['f_statistic']:.2f}",
            f"  Coefficient:         {r['first_stage']['coefficient']:.4f}",
            f"  Weak IV warning:     {'Yes' if r['first_stage']['weak_iv_warning'] else 'No'}",
            "",
            "Rotemberg Diagnostics:",
            f"  Negative weight %:   {r['rotemberg']['negative_weight_share']:.1%}",
            f"  Herfindahl:          {r['rotemberg']['herfindahl']:.3f}",
            f"  Top sector weights:  {r['rotemberg']['top_5_weights'][:3]}",
            "",
            f"Share sum mean:        {r['share_sum_mean']:.3f}",
            "",
            f"Message: {r['message']}",
            "=" * 65,
        ]
        return "\n".join(lines)


def shift_share_iv(
    Y: NDArray[np.floating],
    D: NDArray[np.floating],
    shares: NDArray[np.floating],
    shocks: NDArray[np.floating],
    X: Optional[NDArray[np.floating]] = None,
    clusters: Optional[NDArray] = None,
    inference: Literal["robust", "clustered"] = "robust",
    alpha: float = 0.05,
) -> ShiftShareResult:
    """
    Convenience function for Shift-Share IV estimation.

    Constructs Bartik instrument Z = shares @ shocks and runs 2SLS.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable.
    D : array-like, shape (n,)
        Endogenous treatment.
    shares : array-like, shape (n, S)
        Sector shares (rows should sum to ~1).
    shocks : array-like, shape (S,)
        Aggregate sector shocks.
    X : array-like, shape (n, k), optional
        Control variables.
    clusters : array-like, shape (n,), optional
        Cluster identifiers.
    inference : {'robust', 'clustered'}, default='robust'
        SE type.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    ShiftShareResult
        Estimation results with Rotemberg diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> n, S = 200, 15
    >>> shares = np.random.dirichlet(np.ones(S), n)
    >>> shocks = np.random.normal(0, 0.1, S)
    >>> Z = shares @ shocks
    >>> D = 1.5 * Z + np.random.normal(0, 0.3, n)
    >>> Y = 2.0 * D + np.random.normal(0, 1, n)
    >>> result = shift_share_iv(Y, D, shares, shocks)
    >>> print(f"Estimate: {result['estimate']:.3f}")
    """
    ssiv = ShiftShareIV(inference=inference, alpha=alpha)
    return ssiv.fit(Y, D, shares, shocks, X, clusters)
