#!/usr/bin/python3
#
# Copyright (C) 2019 Google Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd

import openpls.config as c
import openpls.inner_model as im
import openpls.inner_summary as pis
import openpls.outer_model as om
import openpls.weights as w
from openpls.bootstrap import Bootstrap
from openpls.copula import GaussianCopula
from openpls.cta import CTAPLS
from openpls.estimator import Estimator
from openpls.f_squared import FSquared
from openpls.fimix import FIMIX
from openpls.fit import ModelFit
from openpls.fornell_larcker import FornellLarcker
from openpls.higher_order import HigherOrder
from openpls.htmt import HTMT
from openpls.htmt2 import HTMT2
from openpls.ipma import IPMA
from openpls.mga import GroupSpec
from openpls.micom import MICOM
from openpls.plsc import PLSc
from openpls.predict import PLSPredict
from openpls.q_squared import QSquared
from openpls.report import Report
from openpls.scheme import Scheme
from openpls.specific_indirect import specific_indirect_point
from openpls.unidimensionality import Unidimensionality
from openpls.vif import VIF


class Plspm:
    """Estimates path models with latent variables using partial least squares algorithm

    Create an instance of this class in order to estimate a path model using the partial least squares algorithm.
    When the algorithm has performed the calculations to create the estimate, you can then retrieve the inner and outer
    models, scores, the path coefficients, effects, and reliability indicators such as goodness-of-fit
    and unidimensionality. Bootstrapping results can also be retrieved if they were requested.
    """

    def __init__(self, data: pd.DataFrame, config: c.Config, scheme: Scheme = Scheme.CENTROID,
                 iterations: int = 100, tolerance: float = 0.000001, bootstrap: bool = False,
                 bootstrap_iterations: int = 100, processes: int = 2,
                 missing_strategy: str = "casewise"):
        """Creates an instance of the path model calculator.

        Args:
            data: A Pandas DataFrame containing the dataset to be analyzed
            config: An instance of :obj:`.config.Config`
            scheme: The inner weighting scheme to use: :attr:`.Scheme.CENTROID` (default), :attr:`.Scheme.FACTORIAL` or :attr:`.Scheme.PATH` (see documentation for :mod:`.scheme`)
            iterations: The maximum number of iterations to try to get the algorithm to converge (default and minimum 100).
            tolerance: The tolerance criterion for iterations (default 0.000001, must be >0)
            bootstrap: Whether to perform bootstrap validation (default is not to perform validation)
            bootstrap_iterations: The number of bootstrap samples to use if bootstrap validation is enabled (default and minimum 100)
            processes: The number of processes to use while bootstrapping (bootstrap_iterations must be a multiple of processes)
            missing_strategy: How to handle NaN in indicator columns. ``"casewise"``
                (default, matches upstream) drops rows containing any NaN in the
                model's indicators. ``"mean"`` replaces each NaN with the column
                mean of the corresponding indicator (matches the "Mean replacement"
                option in commercial PLS-SEM software).

        Raises:
            Exception: if the algorithm cannot converge, or if the requested configuration could not be calculated
        """

        if iterations < 100:
            iterations = 100
        assert tolerance > 0
        assert scheme in Scheme
        if bootstrap_iterations < 10:
            bootstrap_iterations = 100
        assert processes > 0
        assert bootstrap_iterations % processes == 0
        if missing_strategy not in ("casewise", "mean"):
            raise ValueError(
                f"missing_strategy must be 'casewise' or 'mean', got {missing_strategy!r}"
            )

        if missing_strategy == "mean":
            data = self.__mean_replace(data, config)

        estimator = Estimator(config)
        filtered_data = config.filter(data)
        correction = np.sqrt(filtered_data.shape[0] / (filtered_data.shape[0] - 1))

        calculator = w.WeightsCalculatorFactory(config, iterations, tolerance, correction, scheme)
        final_data, scores, weights = estimator.estimate(calculator, filtered_data)
        config = estimator.config()

        self.__inner_model = im.InnerModel(config.path(), scores)
        self.__outer_model = om.OuterModel(final_data, scores, weights, config.odm(config.path()), self.__inner_model.r_squared())
        self.__inner_summary = pis.InnerSummary(config, self.__inner_model.r_squared(),
                                                self.__inner_model.r_squared_adj(), self.__outer_model.model(),
                                                n_obs=filtered_data.shape[0])
        self.__unidimensionality = Unidimensionality(config, filtered_data, correction)
        self.__model_fit = ModelFit(config, final_data, scores, self.__outer_model.model())
        self.__htmt = HTMT(config, final_data)
        self.__scores = scores
        self.__data = filtered_data
        self.__config = config
        self.__scheme = scheme
        self.__q_squared: QSquared | None = None
        self.__vif: VIF | None = None
        self.__plsc: PLSc | None = None
        self.__htmt2: HTMT2 | None = None
        self.__f_squared: FSquared | None = None
        self.__fornell_larcker: FornellLarcker | None = None
        self.__bootstrap = None
        if bootstrap:
            if (filtered_data.shape[0] < 10):
                raise Exception("Bootstrapping could not be performed, at least 10 observations are required.")
            self.__bootstrap = Bootstrap(config, filtered_data, self.__inner_model, self.__outer_model, calculator,
                                         bootstrap_iterations, processes)

    @staticmethod
    def __mean_replace(data: pd.DataFrame, config: c.Config) -> pd.DataFrame:
        """Replaces NaN with the column mean for every indicator in the model."""
        inds = [ind for lv in config.path().index for ind in config.mvs(lv) if ind in data.columns]
        if not inds:
            return data
        out = data.copy()
        # promote int columns so the mean (float) can be assigned in pandas 2.2+
        out[inds] = out[inds].astype(float)
        means = out[inds].mean(skipna=True)
        out[inds] = out[inds].fillna(means)
        return out

    def scores(self) -> pd.DataFrame:
        """Gets the latent variable scores

        Returns:
            a DataFrame with the latent variable scores, with a column for each latent variable. The index is the same as the index of the data passed in.
        """
        return self.__scores

    def data(self) -> pd.DataFrame:
        """The dataset actually used in the fit (after the configured
        missing-value strategy has been applied)."""
        return self.__data

    def config(self) -> c.Config:
        """The :class:`.config.Config` used in the fit."""
        return self.__config

    def outer_model(self) -> pd.DataFrame:
        """Gets the outer model

        Returns:
            a DataFrame with columns for weight, loading, communality, and redundancy, and a row for each manifest variable
        """
        return self.__outer_model.model()

    def inner_model(self) -> pd.DataFrame:
        """
        Gets the inner model for the endogenous latent variables

        Returns:
            a DataFrame with a row for each latent variable with a direct path to it, and columns for estimate, std error, t, and p>|t|.
        """
        return self.__inner_model.inner_model()

    def path_coefficients(self) -> pd.DataFrame:
        """
        Gets the path coefficient matrix

        Returns:
            a DataFrame of similar form to the Path matrix passed into :class:`openpls.config.Config`, with the relevant path coefficients in each cell
        """
        return self.__inner_model.path_coefficients()

    def crossloadings(self) -> pd.DataFrame:
        """Gets the crossloadings

        Returns:
            a DataFrame with the latent variables as the columns and the manifest variables as the index
        """
        return self.__outer_model.crossloadings()

    def inner_summary(self) -> pd.DataFrame:
        """Gets a summary of the inner model

        Returns:
            a DataFrame with the latent variables as the index, and columns for latent variable type (Exogenous or Endogenous), R squared, block communality, mean redundancy, and AVE (average variance extracted)
        """
        return self.__inner_summary.summary()

    def goodness_of_fit(self) -> float:
        """Gets goodness-of-fit

        Returns:
            goodness-of-fit
        """
        return self.__inner_summary.goodness_of_fit()

    def effects(self) -> pd.DataFrame:
        """Gets direct, indirect, and total effects for each path

        Returns:
            a DataFrame with an entry in the index for every path in the model, and a column for direct, indirect, and total effects for the corresponding path.
        """
        return self.__inner_model.effects()

    def f_squared(self) -> FSquared:
        """Cohen's f² effect size per structural-model edge.

        For every directed edge ``predictor -> endogenous``, refits the
        OLS for the endogenous LV without the predictor and reports
        ``f² = (R²_full - R²_reduced) / (1 - R²_full)`` (Cohen 1988;
        Hair, Hult, Ringle & Sarstedt 2022). Conventional thresholds:
        ``f² >= 0.02`` small, ``>= 0.15`` medium, ``>= 0.35`` large.
        Computed lazily on first call and cached.

        Returns:
            an instance of :class:`.f_squared.FSquared` exposing
            ``table()`` (long-format with effect-size labels) and
            ``matrix()`` (square matrix mirroring the path matrix).
        """
        if self.__f_squared is None:
            self.__f_squared = FSquared(
                self.__config,
                self.__scores,
                self.__inner_model.path_coefficients(),
                self.__inner_model.r_squared(),
            )
        return self.__f_squared

    def fornell_larcker(self) -> FornellLarcker:
        """Fornell-Larcker discriminant-validity criterion (1981).

        Returns a square matrix with ``sqrt(AVE)`` on the diagonal and
        inter-construct correlations off-diagonal. The criterion holds
        for a reflective LV when its diagonal entry exceeds every
        absolute off-diagonal entry in its row. Formative (Mode B) and
        single-indicator LVs receive ``NaN`` on the diagonal because
        AVE is undefined for them; for those constructs use HTMT
        (Henseler, Ringle & Sarstedt 2015) instead. Computed lazily and
        cached.

        Returns:
            an instance of :class:`.fornell_larcker.FornellLarcker`
            exposing ``matrix()``, ``ave()``, and ``summary()``.
        """
        if self.__fornell_larcker is None:
            self.__fornell_larcker = FornellLarcker(
                self.__config,
                self.__scores,
                self.__outer_model.model(),
            )
        return self.__fornell_larcker

    def specific_indirect_effects(
        self,
        source: str,
        target: str,
        through: list[str] | None = None,
    ) -> pd.DataFrame:
        """Point-estimate specific indirect effects (mediation analysis).

        Returns the product of path coefficients along each mediation
        chain ``source -> M1 -> ... -> target``. With ``through=None``
        (default), every indirect chain in the structural model is
        enumerated; with ``through`` set, only that single chain is
        returned. For bootstrap confidence intervals use
        :meth:`~openpls.bootstrap.Bootstrap.specific_indirect_effects`.

        Args:
            source: source LV name.
            target: target LV name.
            through: explicit mediator chain (e.g. ``["M1", "M2"]``) or
                ``None`` to enumerate all chains.

        Returns:
            DataFrame indexed by chain label (e.g. ``"A -> M -> Y"``)
            with columns ``from``, ``to``, ``via`` (tuple of intermediates),
            and ``estimate``.

        Raises:
            ValueError: if ``source == target``, if no indirect chain
                exists, or if ``through`` references nonexistent edges.
            KeyError: if ``source`` or ``target`` is not in the model.
        """
        path = (self.path_coefficients() != 0).astype(int)
        return specific_indirect_point(
            self.path_coefficients(),
            path,
            source,
            target,
            through=through,
        )

    def report(self, include_rho_a: bool = True, include_htmt2: bool = True) -> Report:
        """Publication-ready summary report bundling the standard PLS-SEM panels.

        Aggregates reliability (Cronbach alpha, rho_A, rho_C, AVE),
        discriminant validity (HTMT, optional HTMT2, Fornell-Larcker),
        structural paths with Cohen's f² and effect-size labels,
        per-LV R² / adjusted R² / BIC, fit indices (SRMR, d_ULS, GoF),
        and collinearity (outer + inner VIF) into one object. Use
        :meth:`.report.Report.to_dict` to bundle every section for
        export (e.g. JSON). Underlying values come from the existing
        lazy-cached methods on this class, so calling ``report()``
        multiple times is cheap.

        Args:
            include_rho_a: include the Dijkstra-Henseler rho_A column in
                the reliability table (computed via PLSc). Default True.
            include_htmt2: include the HTMT2 (geometric-mean) matrix in
                the discriminant-validity panel. Default True.

        Returns:
            a :class:`.report.Report` instance.
        """
        return Report(self, include_rho_a=include_rho_a, include_htmt2=include_htmt2)

    def unidimensionality(self) -> pd.DataFrame:
        """Gets the results of checking the unidimensionality of blocks (only meaningful for reflective / mode A blocks)

        Returns:
            a DataFrame with the latent variables as the index, and columns for Cronbach's Alpha, Dillon-Goldstein Rho, and the eigenvalues of the first and second principal components.
        """
        return self.__unidimensionality.summary()

    def htmt(self) -> HTMT:
        """Gets the Heterotrait-Monotrait Ratio of Correlations.

        Returns:
            an instance of :class:`.htmt.HTMT` from which the HTMT matrix
            and a long-format pair list can be retrieved.
        """
        return self.__htmt

    def htmt2(self) -> HTMT2:
        """Gets HTMT2 — the geometric-mean refinement of HTMT
        (Roemer, Schuberth & Henseler, 2021).

        HTMT2 replaces the two arithmetic means in HTMT with geometric
        means; it is consistent under congeneric measurement and removes
        the bias HTMT shows when indicator loadings within a block are
        unequal. Computed lazily on first call.

        Returns:
            an instance of :class:`.htmt2.HTMT2` exposing ``matrix()``
            and ``pairs()``.
        """
        if self.__htmt2 is None:
            self.__htmt2 = HTMT2(self.__config, self.__data)
        return self.__htmt2

    def model_fit(self) -> ModelFit:
        """Gets the model-fit indices (SRMR, d_ULS).

        Returns:
            an instance of :class:`.fit.ModelFit` from which the saturated
            SRMR, d_ULS and the indicator-residual matrix can be retrieved.
        """
        return self.__model_fit

    def q_squared(self, omission_distance: int = 7) -> pd.DataFrame:
        """Gets Stone-Geisser Q² (cross-validated redundancy) per endogenous LV.

        Computed lazily via blindfolding with the given omission distance D
        (default 7). Q² > 0 indicates predictive relevance.

        Args:
            omission_distance: blindfolding parameter D. Each round omits every
                D-th row of the target indicators. Must be >= 2.

        Returns:
            a DataFrame indexed by endogenous LV with a single `q_squared` column.
        """
        if self.__q_squared is None or self.__q_squared.omission_distance != omission_distance:
            self.__q_squared = QSquared(self.__config, self.__data, self.__scheme, omission_distance)
        return self.__q_squared.values()

    def predict(
        self,
        k: int = 10,
        repeats: int = 1,
        seed: int | None = 42,
        lm_predictor_set: str = "direct",
    ) -> PLSPredict:
        """PLSpredict: out-of-sample predictive power via k-fold CV.

        Per-indicator RMSE and MAE for both PLS and the LM benchmark, plus
        Q²_predict (Shmueli et al. 2019).

        Args:
            k: number of folds (default 10, must be >= 2 and <= n).
            repeats: how many times to shuffle and re-fold (default 1).
            seed: random seed for the fold shuffle. Pass ``None`` for
                non-deterministic.
            lm_predictor_set: which LV block feeds the LM benchmark.
                ``"direct"`` (default) uses the LV's direct path predecessors;
                ``"earliest_antecedents"`` walks upstream through every
                mediator and uses only the exogenous LVs at the top of the
                DAG (Shmueli/Hair/Ringle 2019 convention, matches SmartPLS).

        Returns:
            a :class:`.predict.PLSPredict` instance. Call ``metrics()`` for
            the per-indicator error table and ``summary()`` for the PLS-vs-LM
            verdict.
        """
        return PLSPredict(
            self.__config,
            self.__data,
            self.__scheme,
            k=k,
            repeats=repeats,
            seed=seed,
            lm_predictor_set=lm_predictor_set,
        )

    def vif(self) -> VIF:
        """Variance Inflation Factor diagnostics for the outer and inner model.

        Computed lazily on first call. ``items()`` returns per-indicator
        VIF within each construct block (collinearity diagnostic, primarily
        for Mode B / formative blocks). ``inner()`` returns per-predictor
        VIF for each endogenous LV (structural collinearity among
        antecedents).

        Returns:
            an instance of :class:`.vif.VIF` exposing ``items()`` and
            ``inner()``.
        """
        if self.__vif is None:
            self.__vif = VIF(self.__config, self.__data, self.__scores)
        return self.__vif

    def plsc(self) -> PLSc:
        """Consistent PLS (Dijkstra & Henseler 2015).

        Corrects path coefficients, loadings, and R² for measurement-error
        attenuation in reflective (Mode A) constructs. Computed lazily on
        first call and cached.

        Returns:
            an instance of :class:`.plsc.PLSc` exposing ``rho_a()``,
            ``path_coefficients()``, ``r_squared()``, ``loadings()``, and
            ``summary()``.
        """
        if self.__plsc is None:
            self.__plsc = PLSc(
                self.__config,
                self.__data,
                self.__scores,
                self.__outer_model.model(),
            )
        return self.__plsc

    def ipma(
        self,
        target: str,
        scale_min: float | None = None,
        scale_max: float | None = None,
        indicator_scales: dict[str, tuple[float, float]] | None = None,
    ) -> IPMA:
        """Importance-Performance Map Analysis for a target endogenous LV.

        Args:
            target: name of the endogenous LV to analyze.
            scale_min, scale_max: common scale bounds (e.g. 1 and 7 for a
                7-point Likert). If both are None, each indicator is rescaled
                from its observed min/max.
            indicator_scales: optional ``{indicator: (min, max)}`` overrides.

        Returns:
            an :class:`.ipma.IPMA` instance. Call ``latent_variables()`` for
            the LV-level table and ``indicators()`` for the indicator-level
            breakdown.
        """
        return IPMA(
            self.__config,
            self.__data,
            self.outer_model(),
            self.effects(),
            target,
            scale_min=scale_min,
            scale_max=scale_max,
            indicator_scales=indicator_scales,
        )

    def cta(
        self,
        n_boot: int = 500,
        alpha: float = 0.05,
        seed: int | None = 42,
    ) -> CTAPLS:
        """Confirmatory Tetrad Analysis for PLS (Gudergan et al. 2008).

        Tests whether reflective (Mode A) measurement is supported for each
        block with at least four indicators. Each block's tetrads are
        bootstrapped under ``H0: tau = 0`` and reduced with a within-block
        Holm step-down correction at ``alpha``.

        Args:
            n_boot: number of bootstrap resamples per block (default 500,
                must be >= 50).
            alpha: family-wise significance level for the Holm correction
                (default 0.05, must be in ``(0, 1)``).
            seed: RNG seed for the bootstrap. Pass ``None`` for
                non-deterministic results.

        Returns:
            a :class:`.cta.CTAPLS` instance. Call ``tetrads()`` for the
            per-tetrad table and ``summary()`` for the per-block verdict.
        """
        return CTAPLS(
            self.__config,
            self.__data,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )

    def copula(
        self,
        endogenous: str,
        suspected: list[str] | None = None,
        n_boot: int = 500,
        seed: int | None = 42,
    ) -> GaussianCopula:
        """Gaussian-copula endogeneity test (Park & Gupta 2012;
        Hult, Hair, Proksch, Sarstedt, Pinkwart & Ringle 2018).

        For the structural equation of ``endogenous``, augments each
        suspected predecessor LV with a copula term
        ``P_k = Phi^{-1}(F_n(X_k))`` and tests its OLS coefficient by
        non-parametric bootstrap. A significant copula coefficient
        indicates endogeneity in that predictor.

        Args:
            endogenous: name of the endogenous LV whose structural
                equation is tested.
            suspected: predecessor LVs to augment with a copula term. If
                ``None``, all predecessors are tested.
            n_boot: number of bootstrap resamples (default 500, must be
                >= 50).
            seed: RNG seed for the bootstrap. ``None`` for
                non-deterministic.

        Returns:
            a :class:`.copula.GaussianCopula` instance. Call
            ``coefficients()`` for the per-predictor diagnostics,
            ``augmented_paths()`` for the endogeneity-corrected
            structural estimates, and ``summary()`` for a
            per-predictor verdict (including a Cramér-von Mises
            admissibility check).
        """
        return GaussianCopula(
            self.__config,
            self.__scores,
            endogenous,
            suspected=suspected,
            n_boot=n_boot,
            seed=seed,
        )

    def fimix(
        self,
        n_classes: int,
        max_iter: int = 500,
        tolerance: float = 1e-6,
        n_restarts: int = 5,
        seed: int | None = 42,
    ) -> FIMIX:
        """Finite Mixture PLS for latent class segmentation.

        Runs FIMIX-PLS (Hahn et al. 2002) on the fitted model's LV scores.
        Detects ``n_classes`` subgroups sharing the measurement model but
        with distinct structural-path coefficients.

        Args:
            n_classes: number of mixture components K (must be >= 2).
            max_iter: maximum EM iterations per restart (default 500).
            tolerance: convergence tolerance on the log-likelihood.
            n_restarts: number of random EM restarts; the best is kept.
            seed: RNG seed for restart initialization. ``None`` for
                non-deterministic.

        Returns:
            a :class:`.fimix.FIMIX` instance. Call ``memberships()``,
            ``class_paths()``, and ``fit_criteria()`` for the per-class
            results and information criteria for model selection.
        """
        return FIMIX(
            self,
            n_classes=n_classes,
            max_iter=max_iter,
            tolerance=tolerance,
            n_restarts=n_restarts,
            seed=seed,
        )

    def higher_order(
        self,
        name: str,
        first_order: list[str],
        mode,
        structure,
        iterations: int = 100,
        tolerance: float = 1e-6,
        missing_strategy: str = "casewise",
    ) -> HigherOrder:
        """Disjoint two-stage higher-order construct
        (Sarstedt et al. 2019; Hair et al. 2022).

        Uses this fit as **stage 1** and fits a **stage-2** model in
        which ``name`` is a new construct whose indicators are the
        first-order LV scores produced by stage 1. The four canonical
        HOC types (R-R, R-F, F-R, F-F) follow from the existing first-
        order modes in this config plus the ``mode`` chosen here.

        Args:
            name: name of the second-order construct. Must not collide
                with an existing LV or data column.
            first_order: first-order LVs (from this fit) to roll into
                the HOC. At least two are required.
            mode: measurement mode of the HOC w.r.t. its first-order
                indicators (:class:`.mode.Mode`).
            structure: :class:`.config.Structure` of the stage-2 path
                model. Must include ``name`` and must NOT include any
                of the ``first_order`` LVs.
            iterations, tolerance, missing_strategy: stage-2 ``Plspm``
                fit options.

        Returns:
            a :class:`.higher_order.HigherOrder` instance exposing the
            stage-2 fit (``refit()``) and HOC-specific accessors
            (``loadings()``, ``summary()``).
        """
        return HigherOrder(
            self,
            name=name,
            first_order=first_order,
            mode=mode,
            structure=structure,
            scheme=self.__scheme,
            iterations=iterations,
            tolerance=tolerance,
            missing_strategy=missing_strategy,
        )

    def micom(
        self,
        data: pd.DataFrame,
        grouping_column: str,
        group_a: GroupSpec,
        group_b: GroupSpec,
        iterations: int = 1000,
        seed: int | None = 42,
    ) -> MICOM:
        """Measurement Invariance of Composite Models (Henseler, Ringle & Sarstedt 2016).

        Three-step assessment that must be completed before comparing
        composite scores across two groups (e.g. before MGA or moderation):

        - **Step 1** Configural invariance: same indicators, same data
          treatment, same algorithm settings (always satisfied here by
          reusing this fit's :class:`~openpls.config.Config`).
        - **Step 2** Compositional invariance: per-construct correlation
          ``c`` between group-A and group-B composite weights, tested
          against ``c = 1`` via permutation.
        - **Step 3** Equality of composite means and variances: tested
          via permutation on pooled-weight composite scores.

        Args:
            data: full dataset including the grouping column (the
                grouping column is not required to be part of the model).
            grouping_column: name of the column in ``data`` that
                distinguishes the two groups.
            group_a, group_b: :class:`~openpls.mga.GroupSpec` describing
                each subset (by categorical ``values`` or numeric ``range``).
            iterations: number of permutation iterations for Steps 2 and 3
                (default 1000).
            seed: RNG seed. Pass ``None`` for non-deterministic.

        Returns:
            a :class:`~openpls.micom.MICOM` instance exposing ``step2()``,
            ``step3()`` and ``summary()`` per construct.
        """
        return MICOM(
            data,
            self.__config,
            grouping_column=grouping_column,
            group_a=group_a,
            group_b=group_b,
            scheme=self.__scheme,
            iterations=iterations,
            seed=seed,
        )

    def bootstrap(self) -> Bootstrap:
        """Gets the results of bootstrap validation, if requested

        Returns:
            an instance of :class:`.bootstrap.Bootstrap` which can be queried for bootstrapping results

        Raises:
            Exception: if bootstrap validation was not requested or if there were insufficient (<10) observations
        """
        if self.__bootstrap is None:
            raise Exception("To perform bootstrap validation, set the parameter bootstrap to True when calling Plspm")
        return self.__bootstrap
