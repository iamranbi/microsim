"""Regression model containers and the bridge classes that evaluate them against a Person.

This package groups two layers of the "coefficients -> person -> risk" machinery:

- Data containers (``RegressionModel``, ``CoxRegressionModel``): passive holders of
  regression coefficients, standard errors, and residual/hazard parameters loaded
  from the ``data/*ModelSpec.json`` files.
- Bridge classes (``LinearRiskFactorModel`` and its subclasses): wrap a container, read
  a Person's attributes, and compute the next risk-factor value or outcome risk (linear,
  logistic, Cox, relative risk, linear-probability, and rounded-linear variants).

    from microsim.regression_models import LinearRiskFactorModel, RegressionModel
"""

from microsim.regression_models.regression_model import RegressionModel
from microsim.regression_models.cox_regression_model import CoxRegressionModel
from microsim.regression_models.linear_risk_factor_model import LinearRiskFactorModel
from microsim.regression_models.logistic_risk_factor_model import LogisticRiskFactorModel
from microsim.regression_models.rand_intercept_logistic_risk_factor_model import (
    RandInterceptLogisticRiskFactorModel,
)
from microsim.regression_models.cox_risk_factor_model import CoxRiskFactorModel
from microsim.regression_models.relative_risk_factor_model import RelativeRiskFactorModel
from microsim.regression_models.linear_probability_risk_factor_model import (
    LinearProbabilityRiskFactorModel,
)
from microsim.regression_models.rounded_linear_risk_factor_model import (
    RoundedLinearRiskFactorModel,
)

__all__ = [
    "RegressionModel",
    "CoxRegressionModel",
    "LinearRiskFactorModel",
    "LogisticRiskFactorModel",
    "RandInterceptLogisticRiskFactorModel",
    "CoxRiskFactorModel",
    "RelativeRiskFactorModel",
    "LinearProbabilityRiskFactorModel",
    "RoundedLinearRiskFactorModel",
]
