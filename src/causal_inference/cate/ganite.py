"""GANITE: Estimation of Individualized Treatment Effects using GANs.

This module implements GANITE (Yoon et al., ICLR 2018), a GAN-based approach
for counterfactual outcome generation and individualized treatment effect estimation.

Architecture
------------
GANITE consists of two adversarial blocks:

1. Counterfactual Block: Generates missing potential outcomes
   - Generator G: (X, T, Y_factual, noise) -> Y_counterfactual
   - Discriminator D_cf: Classifies real vs generated outcomes

2. ITE Block: Refines treatment effect estimates
   - Generator I: (X, Y0_hat, Y1_hat) -> ITE
   - Discriminator D_ite: Quality discrimination

References
----------
- Yoon et al. (2018). "GANITE: Estimation of Individualized Treatment Effects
  using Generative Adversarial Nets." ICLR.
"""

from typing import Tuple

import numpy as np
from scipy import stats

from .base import CATEResult, validate_cate_inputs


def _check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


if _check_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    # =============================================================================
    # PyTorch Model Classes
    # =============================================================================

    class GANITECounterfactualGenerator(nn.Module):
        """Generator for counterfactual outcomes.

        Takes factual data and noise, generates counterfactual outcome.

        Parameters
        ----------
        input_dim : int
            Dimension of covariates X.
        hidden_dim : int
            Hidden layer dimension.
        noise_dim : int
            Dimension of noise vector.
        """

        def __init__(self, input_dim: int, hidden_dim: int, noise_dim: int):
            super().__init__()
            # Input: X (p) + T (1) + Y (1) + noise (noise_dim)
            self.fc1 = nn.Linear(input_dim + 2 + noise_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(
            self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, noise: torch.Tensor
        ) -> torch.Tensor:
            """Generate counterfactual outcome.

            Parameters
            ----------
            x : torch.Tensor
                Covariates, shape (batch, p).
            t : torch.Tensor
                Treatment, shape (batch, 1).
            y : torch.Tensor
                Factual outcome, shape (batch, 1).
            noise : torch.Tensor
                Random noise, shape (batch, noise_dim).

            Returns
            -------
            torch.Tensor
                Generated counterfactual, shape (batch, 1).
            """
            inp = torch.cat([x, t, y, noise], dim=1)
            h = F.elu(self.fc1(inp))
            h = F.elu(self.fc2(h))
            return self.fc3(h)

    class GANITECounterfactualDiscriminator(nn.Module):
        """Discriminator for counterfactual outcomes.

        Classifies which outcome (Y0 or Y1) is the real factual observation.

        Parameters
        ----------
        input_dim : int
            Dimension of covariates X.
        hidden_dim : int
            Hidden layer dimension.
        """

        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            # Input: X (p) + Y0 (1) + Y1 (1)
            self.fc1 = nn.Linear(input_dim + 2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
            """Discriminate real vs generated outcomes.

            Parameters
            ----------
            x : torch.Tensor
                Covariates, shape (batch, p).
            y0 : torch.Tensor
                Control outcome, shape (batch, 1).
            y1 : torch.Tensor
                Treated outcome, shape (batch, 1).

            Returns
            -------
            torch.Tensor
                Probability that T=1 (Y1 is real), shape (batch, 1).
            """
            inp = torch.cat([x, y0, y1], dim=1)
            h = F.elu(self.fc1(inp))
            h = F.elu(self.fc2(h))
            return torch.sigmoid(self.fc3(h))

    class GANITEITEGenerator(nn.Module):
        """Generator for ITE estimates.

        Takes imputed outcomes and generates refined ITE.

        Parameters
        ----------
        input_dim : int
            Dimension of covariates X.
        hidden_dim : int
            Hidden layer dimension.
        """

        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            # Input: X (p) + Y0_hat (1) + Y1_hat (1)
            self.fc1 = nn.Linear(input_dim + 2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(
            self, x: torch.Tensor, y0_hat: torch.Tensor, y1_hat: torch.Tensor
        ) -> torch.Tensor:
            """Generate ITE estimate.

            Parameters
            ----------
            x : torch.Tensor
                Covariates, shape (batch, p).
            y0_hat : torch.Tensor
                Imputed control outcome, shape (batch, 1).
            y1_hat : torch.Tensor
                Imputed treated outcome, shape (batch, 1).

            Returns
            -------
            torch.Tensor
                ITE estimate, shape (batch, 1).
            """
            inp = torch.cat([x, y0_hat, y1_hat], dim=1)
            h = F.elu(self.fc1(inp))
            h = F.elu(self.fc2(h))
            return self.fc3(h)

    class GANITEITEDiscriminator(nn.Module):
        """Discriminator for ITE quality.

        Discriminates between good and poor ITE estimates.

        Parameters
        ----------
        input_dim : int
            Dimension of covariates X.
        hidden_dim : int
            Hidden layer dimension.
        """

        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            # Input: X (p) + Y0_hat (1) + Y1_hat (1) + ITE (1)
            self.fc1 = nn.Linear(input_dim + 3, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(
            self, x: torch.Tensor, y0_hat: torch.Tensor, y1_hat: torch.Tensor, ite: torch.Tensor
        ) -> torch.Tensor:
            """Discriminate ITE quality.

            Parameters
            ----------
            x : torch.Tensor
                Covariates, shape (batch, p).
            y0_hat : torch.Tensor
                Imputed control outcome, shape (batch, 1).
            y1_hat : torch.Tensor
                Imputed treated outcome, shape (batch, 1).
            ite : torch.Tensor
                ITE estimate, shape (batch, 1).

            Returns
            -------
            torch.Tensor
                Quality score, shape (batch, 1).
            """
            inp = torch.cat([x, y0_hat, y1_hat, ite], dim=1)
            h = F.elu(self.fc1(inp))
            h = F.elu(self.fc2(h))
            return torch.sigmoid(self.fc3(h))

    # =============================================================================
    # GANITE Model Class
    # =============================================================================

    class GANITE:
        """GANITE model for individualized treatment effect estimation.

        Parameters
        ----------
        hidden_dim : int, default=128
            Hidden layer dimension for all networks.
        noise_dim : int, default=32
            Dimension of noise vector for generator.
        alpha : float, default=1.0
            Weight for counterfactual block adversarial loss.
        beta : float, default=1.0
            Weight for ITE block adversarial loss.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=64
            Batch size for training.
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer.
        cf_warmup : int, default=20
            Epochs to train counterfactual block before ITE block.
        random_state : int, default=42
            Random seed for reproducibility.
        """

        def __init__(
            self,
            hidden_dim: int = 128,
            noise_dim: int = 32,
            alpha: float = 1.0,
            beta: float = 1.0,
            epochs: int = 100,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            cf_warmup: int = 20,
            random_state: int = 42,
        ):
            self.hidden_dim = hidden_dim
            self.noise_dim = noise_dim
            self.alpha = alpha
            self.beta = beta
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.cf_warmup = cf_warmup
            self.random_state = random_state

            self._fitted = False
            self.input_dim = None

            # Models (initialized in fit)
            self.cf_generator = None
            self.cf_discriminator = None
            self.ite_generator = None
            self.ite_discriminator = None

        def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "GANITE":
            """Fit GANITE model.

            Parameters
            ----------
            X : np.ndarray
                Covariates, shape (n, p).
            T : np.ndarray
                Binary treatment, shape (n,).
            Y : np.ndarray
                Outcomes, shape (n,).

            Returns
            -------
            self
                Fitted model.
            """
            if not _check_torch_available():
                raise ImportError("PyTorch is required for GANITE. Install with: pip install torch")

            # Set random seed
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            # Ensure 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n, p = X.shape
            self.input_dim = p

            # Convert to tensors
            X_t = torch.FloatTensor(X)
            T_t = torch.FloatTensor(T.reshape(-1, 1))
            Y_t = torch.FloatTensor(Y.reshape(-1, 1))

            # Initialize models
            self.cf_generator = GANITECounterfactualGenerator(p, self.hidden_dim, self.noise_dim)
            self.cf_discriminator = GANITECounterfactualDiscriminator(p, self.hidden_dim)
            self.ite_generator = GANITEITEGenerator(p, self.hidden_dim)
            self.ite_discriminator = GANITEITEDiscriminator(p, self.hidden_dim)

            # Optimizers
            opt_cf_g = torch.optim.Adam(self.cf_generator.parameters(), lr=self.learning_rate)
            opt_cf_d = torch.optim.Adam(self.cf_discriminator.parameters(), lr=self.learning_rate)
            opt_ite_g = torch.optim.Adam(self.ite_generator.parameters(), lr=self.learning_rate)
            opt_ite_d = torch.optim.Adam(self.ite_discriminator.parameters(), lr=self.learning_rate)

            # Create dataloader
            dataset = TensorDataset(X_t, T_t, Y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop
            for epoch in range(self.epochs):
                for batch_x, batch_t, batch_y in loader:
                    batch_size = batch_x.size(0)

                    # ============================================================
                    # Phase 1: Train Counterfactual Block
                    # ============================================================

                    # Generate noise
                    noise = torch.randn(batch_size, self.noise_dim)

                    # Generate counterfactual
                    y_cf = self.cf_generator(batch_x, batch_t, batch_y, noise)

                    # Construct Y0, Y1 (combine factual and counterfactual)
                    # If T=0: Y0 = Y_factual, Y1 = Y_cf
                    # If T=1: Y1 = Y_factual, Y0 = Y_cf
                    y0 = (1 - batch_t) * batch_y + batch_t * y_cf
                    y1 = batch_t * batch_y + (1 - batch_t) * y_cf

                    # --- Train Discriminator ---
                    opt_cf_d.zero_grad()
                    d_out = self.cf_discriminator(batch_x, y0.detach(), y1.detach())

                    # Labels: T=1 means Y1 is real
                    d_loss = F.binary_cross_entropy(d_out, batch_t)
                    d_loss.backward()
                    opt_cf_d.step()

                    # --- Train Generator ---
                    opt_cf_g.zero_grad()

                    # Regenerate with fresh noise
                    noise = torch.randn(batch_size, self.noise_dim)
                    y_cf = self.cf_generator(batch_x, batch_t, batch_y, noise)
                    y0 = (1 - batch_t) * batch_y + batch_t * y_cf
                    y1 = batch_t * batch_y + (1 - batch_t) * y_cf

                    # Adversarial loss (fool discriminator)
                    d_out = self.cf_discriminator(batch_x, y0, y1)
                    g_adv_loss = F.binary_cross_entropy(d_out, 1 - batch_t)

                    # Reconstruction loss on factual
                    # factual = T*Y1 + (1-T)*Y0 should equal Y
                    y_recon = batch_t * y1 + (1 - batch_t) * y0
                    recon_loss = F.mse_loss(y_recon, batch_y)

                    g_loss = recon_loss + self.alpha * g_adv_loss
                    g_loss.backward()
                    opt_cf_g.step()

                    # ============================================================
                    # Phase 2: Train ITE Block (after warmup)
                    # ============================================================

                    if epoch >= self.cf_warmup:
                        # Get imputed outcomes from CF block
                        with torch.no_grad():
                            noise = torch.randn(batch_size, self.noise_dim)
                            y_cf_imp = self.cf_generator(batch_x, batch_t, batch_y, noise)
                            y0_hat = (1 - batch_t) * batch_y + batch_t * y_cf_imp
                            y1_hat = batch_t * batch_y + (1 - batch_t) * y_cf_imp

                        # True ITE from imputed outcomes
                        ite_true = y1_hat - y0_hat

                        # --- Train ITE Generator ---
                        opt_ite_g.zero_grad()
                        ite_pred = self.ite_generator(batch_x, y0_hat, y1_hat)

                        # ITE matching loss
                        ite_loss = F.mse_loss(ite_pred, ite_true)

                        # Adversarial loss
                        d_ite_out = self.ite_discriminator(batch_x, y0_hat, y1_hat, ite_pred)
                        ite_adv_loss = F.binary_cross_entropy(d_ite_out, torch.ones_like(d_ite_out))

                        ite_g_loss = ite_loss + self.beta * ite_adv_loss
                        ite_g_loss.backward()
                        opt_ite_g.step()

                        # --- Train ITE Discriminator ---
                        opt_ite_d.zero_grad()

                        # Real: true ITE from imputation
                        d_real = self.ite_discriminator(batch_x, y0_hat, y1_hat, ite_true)
                        # Fake: generated ITE
                        ite_pred_d = self.ite_generator(batch_x, y0_hat, y1_hat).detach()
                        d_fake = self.ite_discriminator(batch_x, y0_hat, y1_hat, ite_pred_d)

                        d_ite_loss = (
                            F.binary_cross_entropy(d_real, torch.ones_like(d_real))
                            + F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
                        ) / 2
                        d_ite_loss.backward()
                        opt_ite_d.step()

            self._fitted = True
            return self

        def predict_cate(self, X: np.ndarray) -> np.ndarray:
            """Predict CATE for new samples.

            Parameters
            ----------
            X : np.ndarray
                Covariates, shape (n, p) or (n,).

            Returns
            -------
            np.ndarray
                CATE predictions, shape (n,).
            """
            if not self._fitted:
                raise ValueError("Model must be fitted before prediction.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_t = torch.FloatTensor(X)
            n = X_t.size(0)

            # Use ITE generator with zero outcomes (will use learned representation)
            # Actually need to impute Y0, Y1 first using CF block
            self.cf_generator.eval()
            self.ite_generator.eval()

            with torch.no_grad():
                # For prediction, we don't have factual Y
                # Use mean-field approximation: predict both outcomes
                # Predict Y0: pretend T=0
                noise = torch.randn(n, self.noise_dim)
                t0 = torch.zeros(n, 1)
                # Use zero as placeholder Y (will be generated)
                y_placeholder = torch.zeros(n, 1)

                # Generate multiple samples and average
                n_samples = 10
                y0_samples = []
                y1_samples = []

                for _ in range(n_samples):
                    noise = torch.randn(n, self.noise_dim)

                    # Generate Y1 given T=0 (counterfactual)
                    y1_cf = self.cf_generator(X_t, t0, y_placeholder, noise)
                    y1_samples.append(y1_cf)

                    # Generate Y0 given T=1 (counterfactual)
                    t1 = torch.ones(n, 1)
                    y0_cf = self.cf_generator(X_t, t1, y_placeholder, noise)
                    y0_samples.append(y0_cf)

                y0_hat = torch.stack(y0_samples).mean(dim=0)
                y1_hat = torch.stack(y1_samples).mean(dim=0)

                # Use ITE generator for final prediction
                cate = self.ite_generator(X_t, y0_hat, y1_hat)

            return cate.numpy().flatten()

        def predict_y0(self, X: np.ndarray) -> np.ndarray:
            """Predict Y(0) for samples.

            Parameters
            ----------
            X : np.ndarray
                Covariates.

            Returns
            -------
            np.ndarray
                Y(0) predictions.
            """
            if not self._fitted:
                raise ValueError("Model must be fitted before prediction.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_t = torch.FloatTensor(X)
            n = X_t.size(0)

            self.cf_generator.eval()
            with torch.no_grad():
                noise = torch.randn(n, self.noise_dim)
                t1 = torch.ones(n, 1)
                y_placeholder = torch.zeros(n, 1)
                y0 = self.cf_generator(X_t, t1, y_placeholder, noise)

            return y0.numpy().flatten()

        def predict_y1(self, X: np.ndarray) -> np.ndarray:
            """Predict Y(1) for samples.

            Parameters
            ----------
            X : np.ndarray
                Covariates.

            Returns
            -------
            np.ndarray
                Y(1) predictions.
            """
            if not self._fitted:
                raise ValueError("Model must be fitted before prediction.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_t = torch.FloatTensor(X)
            n = X_t.size(0)

            self.cf_generator.eval()
            with torch.no_grad():
                noise = torch.randn(n, self.noise_dim)
                t0 = torch.zeros(n, 1)
                y_placeholder = torch.zeros(n, 1)
                y1 = self.cf_generator(X_t, t0, y_placeholder, noise)

            return y1.numpy().flatten()

    # =============================================================================
    # API Function
    # =============================================================================

    def ganite(
        outcomes: np.ndarray,
        treatment: np.ndarray,
        covariates: np.ndarray,
        hidden_dim: int = 128,
        noise_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        cf_warmup: int = 20,
        alpha_ci: float = 0.05,
        random_state: int = 42,
    ) -> CATEResult:
        """Estimate CATE using GANITE.

        GANITE (Yoon et al., ICLR 2018) uses generative adversarial networks
        to generate counterfactual outcomes and estimate individualized
        treatment effects.

        Parameters
        ----------
        outcomes : np.ndarray
            Outcome variable Y, shape (n,).
        treatment : np.ndarray
            Binary treatment T, shape (n,).
        covariates : np.ndarray
            Covariate matrix X, shape (n, p) or (n,).
        hidden_dim : int, default=128
            Hidden layer dimension for all networks.
        noise_dim : int, default=32
            Dimension of noise vector for generator.
        alpha : float, default=1.0
            Weight for counterfactual block adversarial loss.
        beta : float, default=1.0
            Weight for ITE block adversarial loss.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=64
            Batch size for training.
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer.
        cf_warmup : int, default=20
            Epochs to train counterfactual block before ITE block.
        alpha_ci : float, default=0.05
            Significance level for confidence interval.
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        CATEResult
            Dictionary with cate, ate, ate_se, ci_lower, ci_upper, method.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> n = 500
        >>> X = np.random.randn(n, 5)
        >>> T = np.random.binomial(1, 0.5, n)
        >>> Y = 1 + X[:, 0] + 2 * T + np.random.randn(n)
        >>> result = ganite(Y, T, X, epochs=50)
        >>> abs(result["ate"] - 2.0) < 1.0
        True

        References
        ----------
        Yoon et al. (2018). "GANITE: Estimation of Individualized Treatment
        Effects using Generative Adversarial Nets." ICLR.
        """
        # Validate inputs
        outcomes, treatment, covariates = validate_cate_inputs(outcomes, treatment, covariates)
        n = len(outcomes)

        # Fit model
        model = GANITE(
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            alpha=alpha,
            beta=beta,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            cf_warmup=cf_warmup,
            random_state=random_state,
        )
        model.fit(covariates, treatment, outcomes)

        # Predict CATE
        cate = model.predict_cate(covariates)

        # Compute ATE
        ate = float(np.mean(cate))

        # Compute SE using doubly robust influence function
        # First, get propensity and outcome predictions
        y0_pred = model.predict_y0(covariates)
        y1_pred = model.predict_y1(covariates)

        # Simple propensity estimate (logistic on X)
        from sklearn.linear_model import LogisticRegression

        prop_model = LogisticRegression(random_state=random_state, max_iter=1000)
        prop_model.fit(covariates, treatment)
        propensity = prop_model.predict_proba(covariates)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)

        # Doubly robust influence function
        # psi = (T/e - (1-T)/(1-e)) * (Y - T*y1 - (1-T)*y0) + cate
        residual = outcomes - treatment * y1_pred - (1 - treatment) * y0_pred
        weight = treatment / propensity - (1 - treatment) / (1 - propensity)
        psi = weight * residual + cate

        ate_se = float(np.std(psi, ddof=1) / np.sqrt(n))

        # Ensure SE is positive
        if ate_se < 1e-10:
            ate_se = float(np.std(cate, ddof=1) / np.sqrt(n))
            if ate_se < 1e-10:
                ate_se = 0.1  # Fallback

        # Confidence interval
        z = stats.norm.ppf(1 - alpha_ci / 2)
        ci_lower = ate - z * ate_se
        ci_upper = ate + z * ate_se

        return CATEResult(
            cate=cate,
            ate=ate,
            ate_se=ate_se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="ganite",
        )
