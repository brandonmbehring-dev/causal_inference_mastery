"""TEDVAE: Treatment Effect Estimation with Disentangled VAE.

This module implements TEDVAE (Zhang et al., AAAI 2021), a variational autoencoder
with disentangled latent space for causal effect estimation.

Architecture
------------
TEDVAE learns three disentangled latent factors:

1. zt (Instrumental): Affects treatment assignment only
   - Variables that influence T but not Y directly
   - Analogous to instrumental variables

2. zc (Confounding): Affects both treatment and outcome
   - Common causes of T and Y
   - Key for causal identification

3. zy (Risk/Prognostic): Affects outcome only
   - Risk factors independent of treatment assignment
   - Important for outcome prediction

The disentanglement is achieved through the decoder structure:
- Treatment model: P(T|zt, zc) - uses only zt and zc
- Outcome model: E[Y|zc, zy, T] - uses zc, zy, and T (not zt)
- Reconstruction: P(X|zt, zc, zy) - uses all factors

References
----------
- Zhang et al. (2021). "Treatment Effect Estimation with Disentangled
  Latent Factors." AAAI 2021.
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

    class TEDVAEEncoder(nn.Module):
        """Encoder for a single latent factor.

        Maps covariates X to latent distribution parameters (mu, log_var).
        Uses reparameterization trick for sampling.

        Parameters
        ----------
        input_dim : int
            Dimension of input covariates X.
        hidden_dim : int
            Hidden layer dimension.
        latent_dim : int
            Dimension of latent factor.
        """

        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode X to latent distribution parameters.

            Parameters
            ----------
            x : torch.Tensor
                Covariates, shape (batch, input_dim).

            Returns
            -------
            mu : torch.Tensor
                Mean of latent distribution, shape (batch, latent_dim).
            log_var : torch.Tensor
                Log variance of latent distribution, shape (batch, latent_dim).
            """
            h = F.elu(self.fc1(x))
            h = F.elu(self.fc2(h))
            mu = self.fc_mu(h)
            log_var = self.fc_logvar(h)
            return mu, log_var

        def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            """Sample from latent distribution using reparameterization trick.

            Parameters
            ----------
            mu : torch.Tensor
                Mean of latent distribution.
            log_var : torch.Tensor
                Log variance of latent distribution.

            Returns
            -------
            torch.Tensor
                Sampled latent vector.
            """
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

    class TEDVAEDecoder(nn.Module):
        """Decoder for X reconstruction.

        Reconstructs covariates X from all three latent factors (zt, zc, zy).

        Parameters
        ----------
        output_dim : int
            Dimension of reconstructed X.
        hidden_dim : int
            Hidden layer dimension.
        latent_dim_t : int
            Dimension of instrumental factor zt.
        latent_dim_c : int
            Dimension of confounding factor zc.
        latent_dim_y : int
            Dimension of risk factor zy.
        """

        def __init__(
            self,
            output_dim: int,
            hidden_dim: int,
            latent_dim_t: int,
            latent_dim_c: int,
            latent_dim_y: int,
        ):
            super().__init__()
            total_latent = latent_dim_t + latent_dim_c + latent_dim_y
            self.fc1 = nn.Linear(total_latent, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, zt: torch.Tensor, zc: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
            """Reconstruct X from latent factors.

            Parameters
            ----------
            zt : torch.Tensor
                Instrumental factor, shape (batch, latent_dim_t).
            zc : torch.Tensor
                Confounding factor, shape (batch, latent_dim_c).
            zy : torch.Tensor
                Risk factor, shape (batch, latent_dim_y).

            Returns
            -------
            torch.Tensor
                Reconstructed X, shape (batch, output_dim).
            """
            z = torch.cat([zt, zc, zy], dim=1)
            h = F.elu(self.fc1(z))
            h = F.elu(self.fc2(h))
            return self.fc_out(h)

    class TEDVAETreatmentModel(nn.Module):
        """Treatment prediction model.

        Predicts treatment assignment P(T=1|zt, zc).
        Uses only instrumental and confounding factors.

        Parameters
        ----------
        hidden_dim : int
            Hidden layer dimension.
        latent_dim_t : int
            Dimension of instrumental factor zt.
        latent_dim_c : int
            Dimension of confounding factor zc.
        """

        def __init__(self, hidden_dim: int, latent_dim_t: int, latent_dim_c: int):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim_t + latent_dim_c, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc_out = nn.Linear(hidden_dim // 2, 1)

        def forward(self, zt: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
            """Predict treatment probability.

            Parameters
            ----------
            zt : torch.Tensor
                Instrumental factor, shape (batch, latent_dim_t).
            zc : torch.Tensor
                Confounding factor, shape (batch, latent_dim_c).

            Returns
            -------
            torch.Tensor
                Treatment probability P(T=1), shape (batch, 1).
            """
            z = torch.cat([zt, zc], dim=1)
            h = F.elu(self.fc1(z))
            h = F.elu(self.fc2(h))
            return torch.sigmoid(self.fc_out(h))

    class TEDVAEOutcomeModel(nn.Module):
        """Outcome prediction model.

        Predicts outcome E[Y|zc, zy, T].
        Uses confounding factor, risk factor, and treatment.
        Does NOT use instrumental factor (key for disentanglement).

        Parameters
        ----------
        hidden_dim : int
            Hidden layer dimension.
        latent_dim_c : int
            Dimension of confounding factor zc.
        latent_dim_y : int
            Dimension of risk factor zy.
        """

        def __init__(self, hidden_dim: int, latent_dim_c: int, latent_dim_y: int):
            super().__init__()
            # Input: zc + zy + T (scalar)
            self.fc1 = nn.Linear(latent_dim_c + latent_dim_y + 1, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc_out = nn.Linear(hidden_dim // 2, 1)

        def forward(self, zc: torch.Tensor, zy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Predict outcome.

            Parameters
            ----------
            zc : torch.Tensor
                Confounding factor, shape (batch, latent_dim_c).
            zy : torch.Tensor
                Risk factor, shape (batch, latent_dim_y).
            t : torch.Tensor
                Treatment indicator, shape (batch, 1).

            Returns
            -------
            torch.Tensor
                Predicted outcome, shape (batch, 1).
            """
            inp = torch.cat([zc, zy, t], dim=1)
            h = F.elu(self.fc1(inp))
            h = F.elu(self.fc2(h))
            return self.fc_out(h)

    # =============================================================================
    # TEDVAE Model Class
    # =============================================================================

    class TEDVAE:
        """TEDVAE model for treatment effect estimation with disentanglement.

        Parameters
        ----------
        latent_dim_t : int, default=8
            Dimension of instrumental latent factor zt.
        latent_dim_c : int, default=16
            Dimension of confounding latent factor zc.
        latent_dim_y : int, default=8
            Dimension of risk latent factor zy.
        hidden_dim : int, default=128
            Hidden layer dimension for all networks.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=64
            Batch size for training.
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer.
        beta_kl : float, default=1.0
            Weight for KL divergence term in ELBO.
        random_state : int, default=42
            Random seed for reproducibility.
        """

        def __init__(
            self,
            latent_dim_t: int = 8,
            latent_dim_c: int = 16,
            latent_dim_y: int = 8,
            hidden_dim: int = 128,
            epochs: int = 100,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            beta_kl: float = 1.0,
            random_state: int = 42,
        ):
            self.latent_dim_t = latent_dim_t
            self.latent_dim_c = latent_dim_c
            self.latent_dim_y = latent_dim_y
            self.hidden_dim = hidden_dim
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.beta_kl = beta_kl
            self.random_state = random_state

            self._fitted = False
            self.input_dim = None

            # Models (initialized in fit)
            self.encoder_t = None
            self.encoder_c = None
            self.encoder_y = None
            self.decoder = None
            self.treatment_model = None
            self.outcome_model = None

        def _kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            """Compute KL divergence from standard normal.

            KL(q(z|x) || p(z)) where p(z) = N(0, I).

            Parameters
            ----------
            mu : torch.Tensor
                Mean of approximate posterior.
            log_var : torch.Tensor
                Log variance of approximate posterior.

            Returns
            -------
            torch.Tensor
                KL divergence (scalar).
            """
            return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TEDVAE":
            """Fit TEDVAE model.

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
                raise ImportError("PyTorch is required for TEDVAE. Install with: pip install torch")

            # Set random seed
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            # Ensure 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n, p = X.shape
            self.input_dim = p

            # Standardize X for better VAE training
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0) + 1e-8
            X_norm = (X - self._x_mean) / self._x_std

            # Standardize Y
            self._y_mean = Y.mean()
            self._y_std = Y.std() + 1e-8
            Y_norm = (Y - self._y_mean) / self._y_std

            # Convert to tensors
            X_t = torch.FloatTensor(X_norm)
            T_t = torch.FloatTensor(T.reshape(-1, 1))
            Y_t = torch.FloatTensor(Y_norm.reshape(-1, 1))

            # Initialize models
            self.encoder_t = TEDVAEEncoder(p, self.hidden_dim, self.latent_dim_t)
            self.encoder_c = TEDVAEEncoder(p, self.hidden_dim, self.latent_dim_c)
            self.encoder_y = TEDVAEEncoder(p, self.hidden_dim, self.latent_dim_y)
            self.decoder = TEDVAEDecoder(
                p, self.hidden_dim, self.latent_dim_t, self.latent_dim_c, self.latent_dim_y
            )
            self.treatment_model = TEDVAETreatmentModel(
                self.hidden_dim, self.latent_dim_t, self.latent_dim_c
            )
            self.outcome_model = TEDVAEOutcomeModel(
                self.hidden_dim, self.latent_dim_c, self.latent_dim_y
            )

            # Collect all parameters
            all_params = (
                list(self.encoder_t.parameters())
                + list(self.encoder_c.parameters())
                + list(self.encoder_y.parameters())
                + list(self.decoder.parameters())
                + list(self.treatment_model.parameters())
                + list(self.outcome_model.parameters())
            )
            optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

            # Create dataloader
            dataset = TensorDataset(X_t, T_t, Y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training loop
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_x, batch_t, batch_y in loader:
                    optimizer.zero_grad()

                    # Encode to all three latent spaces
                    mu_t, logvar_t = self.encoder_t(batch_x)
                    mu_c, logvar_c = self.encoder_c(batch_x)
                    mu_y, logvar_y = self.encoder_y(batch_x)

                    # Sample using reparameterization
                    zt = self.encoder_t.reparameterize(mu_t, logvar_t)
                    zc = self.encoder_c.reparameterize(mu_c, logvar_c)
                    zy = self.encoder_y.reparameterize(mu_y, logvar_y)

                    # Reconstruction loss
                    x_recon = self.decoder(zt, zc, zy)
                    recon_loss = F.mse_loss(x_recon, batch_x)

                    # Treatment prediction loss (BCE)
                    t_pred = self.treatment_model(zt, zc)
                    treatment_loss = F.binary_cross_entropy(t_pred, batch_t)

                    # Outcome prediction loss (MSE)
                    y_pred = self.outcome_model(zc, zy, batch_t)
                    outcome_loss = F.mse_loss(y_pred, batch_y)

                    # KL divergence for all three latent spaces
                    kl_t = self._kl_divergence(mu_t, logvar_t)
                    kl_c = self._kl_divergence(mu_c, logvar_c)
                    kl_y = self._kl_divergence(mu_y, logvar_y)
                    kl_loss = kl_t + kl_c + kl_y

                    # Total ELBO loss
                    loss = recon_loss + treatment_loss + outcome_loss + self.beta_kl * kl_loss

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            self._fitted = True
            return self

        def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Encode X to disentangled latent factors.

            Parameters
            ----------
            X : np.ndarray
                Covariates, shape (n, p).

            Returns
            -------
            zt : np.ndarray
                Instrumental factor, shape (n, latent_dim_t).
            zc : np.ndarray
                Confounding factor, shape (n, latent_dim_c).
            zy : np.ndarray
                Risk factor, shape (n, latent_dim_y).
            """
            if not self._fitted:
                raise ValueError("Model must be fitted before encoding.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Normalize
            X_norm = (X - self._x_mean) / self._x_std
            X_t = torch.FloatTensor(X_norm)

            self.encoder_t.eval()
            self.encoder_c.eval()
            self.encoder_y.eval()

            with torch.no_grad():
                mu_t, _ = self.encoder_t(X_t)
                mu_c, _ = self.encoder_c(X_t)
                mu_y, _ = self.encoder_y(X_t)

            return mu_t.numpy(), mu_c.numpy(), mu_y.numpy()

        def predict_cate(self, X: np.ndarray) -> np.ndarray:
            """Predict CATE for samples.

            CATE = E[Y|zc, zy, T=1] - E[Y|zc, zy, T=0]

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

            # Normalize
            X_norm = (X - self._x_mean) / self._x_std
            X_t = torch.FloatTensor(X_norm)
            n = X_t.size(0)

            self.encoder_c.eval()
            self.encoder_y.eval()
            self.outcome_model.eval()

            with torch.no_grad():
                # Encode to confounding and risk factors (not instrumental)
                mu_c, _ = self.encoder_c(X_t)
                mu_y, _ = self.encoder_y(X_t)

                # Predict Y(1) and Y(0)
                t1 = torch.ones(n, 1)
                t0 = torch.zeros(n, 1)

                y1_norm = self.outcome_model(mu_c, mu_y, t1)
                y0_norm = self.outcome_model(mu_c, mu_y, t0)

                # CATE in normalized scale
                cate_norm = y1_norm - y0_norm

            # Denormalize (only scale, not shift for difference)
            cate = cate_norm.numpy().flatten() * self._y_std

            return cate

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

            X_norm = (X - self._x_mean) / self._x_std
            X_t = torch.FloatTensor(X_norm)
            n = X_t.size(0)

            self.encoder_c.eval()
            self.encoder_y.eval()
            self.outcome_model.eval()

            with torch.no_grad():
                mu_c, _ = self.encoder_c(X_t)
                mu_y, _ = self.encoder_y(X_t)
                t0 = torch.zeros(n, 1)
                y0_norm = self.outcome_model(mu_c, mu_y, t0)

            return y0_norm.numpy().flatten() * self._y_std + self._y_mean

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

            X_norm = (X - self._x_mean) / self._x_std
            X_t = torch.FloatTensor(X_norm)
            n = X_t.size(0)

            self.encoder_c.eval()
            self.encoder_y.eval()
            self.outcome_model.eval()

            with torch.no_grad():
                mu_c, _ = self.encoder_c(X_t)
                mu_y, _ = self.encoder_y(X_t)
                t1 = torch.ones(n, 1)
                y1_norm = self.outcome_model(mu_c, mu_y, t1)

            return y1_norm.numpy().flatten() * self._y_std + self._y_mean

        def predict_propensity(self, X: np.ndarray) -> np.ndarray:
            """Predict propensity scores P(T=1|X).

            Parameters
            ----------
            X : np.ndarray
                Covariates.

            Returns
            -------
            np.ndarray
                Propensity scores.
            """
            if not self._fitted:
                raise ValueError("Model must be fitted before prediction.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_norm = (X - self._x_mean) / self._x_std
            X_t = torch.FloatTensor(X_norm)

            self.encoder_t.eval()
            self.encoder_c.eval()
            self.treatment_model.eval()

            with torch.no_grad():
                mu_t, _ = self.encoder_t(X_t)
                mu_c, _ = self.encoder_c(X_t)
                propensity = self.treatment_model(mu_t, mu_c)

            return propensity.numpy().flatten()

    # =============================================================================
    # API Function
    # =============================================================================

    def tedvae(
        outcomes: np.ndarray,
        treatment: np.ndarray,
        covariates: np.ndarray,
        latent_dim_t: int = 8,
        latent_dim_c: int = 16,
        latent_dim_y: int = 8,
        hidden_dim: int = 128,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        beta_kl: float = 1.0,
        alpha_ci: float = 0.05,
        random_state: int = 42,
    ) -> CATEResult:
        """Estimate CATE using TEDVAE (Treatment Effect Disentangled VAE).

        TEDVAE (Zhang et al., AAAI 2021) uses a variational autoencoder with
        disentangled latent factors to separate instrumental, confounding,
        and prognostic variables for robust treatment effect estimation.

        Parameters
        ----------
        outcomes : np.ndarray
            Outcome variable Y, shape (n,).
        treatment : np.ndarray
            Binary treatment T, shape (n,).
        covariates : np.ndarray
            Covariate matrix X, shape (n, p) or (n,).
        latent_dim_t : int, default=8
            Dimension of instrumental latent factor zt.
        latent_dim_c : int, default=16
            Dimension of confounding latent factor zc.
        latent_dim_y : int, default=8
            Dimension of risk latent factor zy.
        hidden_dim : int, default=128
            Hidden layer dimension for all networks.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=64
            Batch size for training.
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer.
        beta_kl : float, default=1.0
            Weight for KL divergence term in ELBO.
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
        >>> result = tedvae(Y, T, X, epochs=50)
        >>> abs(result["ate"] - 2.0) < 1.5
        True

        Notes
        -----
        **Disentanglement Architecture**:

        TEDVAE learns three latent factors with distinct causal roles:

        - zt (Instrumental): Affects T only, not Y directly
        - zc (Confounding): Affects both T and Y (common causes)
        - zy (Risk): Affects Y only, not T

        This is enforced through the model structure:
        - Treatment model: P(T=1|zt, zc) - no zy
        - Outcome model: E[Y|zc, zy, T] - no zt

        **ELBO Loss**:

        L = L_recon(X) + L_treatment(T) + L_outcome(Y) + beta * (KL_t + KL_c + KL_y)

        References
        ----------
        Zhang et al. (2021). "Treatment Effect Estimation with Disentangled
        Latent Factors." AAAI 2021.
        """
        # Validate inputs
        outcomes, treatment, covariates = validate_cate_inputs(outcomes, treatment, covariates)
        n = len(outcomes)

        # Fit model
        model = TEDVAE(
            latent_dim_t=latent_dim_t,
            latent_dim_c=latent_dim_c,
            latent_dim_y=latent_dim_y,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta_kl=beta_kl,
            random_state=random_state,
        )
        model.fit(covariates, treatment, outcomes)

        # Predict CATE
        cate = model.predict_cate(covariates)

        # Compute ATE
        ate = float(np.mean(cate))

        # Compute SE using doubly robust influence function
        y0_pred = model.predict_y0(covariates)
        y1_pred = model.predict_y1(covariates)

        # Get propensity from model
        propensity = model.predict_propensity(covariates)
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
            method="tedvae",
        )
