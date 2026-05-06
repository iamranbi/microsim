import math
import unittest

from scipy.special import expit

from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.outcome_prevalence_base import OutcomePrevalenceBase
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository
from microsim.outcomes.epilepsy_model import EpilepsyPrevalenceModel
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.person_filter_factory import PersonFilterFactory
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType


def _adults_filter():
    pf = PersonFilterFactory.get_person_filter(addCommonFilters=False)
    pf.add_filter("df", "adults", lambda x: x[DynamicRiskFactorsType.AGE.value] >= 18)
    return pf


class _StubPrevalenceModel(OutcomePrevalenceBase):
    """Minimal subclass with a fixed linear predictor so we can exercise the base-class
    odds-shift formula without depending on real coefficients."""

    _outcomeType = OutcomeType.CARDIOVASCULAR

    def __init__(self, linearPredictor, riskScaling=1.0):
        self._lp = linearPredictor
        self._riskScaling = riskScaling

    def get_linear_predictor_for_person(self, person):
        return self._lp


class TestOutcomePrevalenceBaseOddsShift(unittest.TestCase):
    def test_default_scaling_is_a_no_op(self):
        model = _StubPrevalenceModel(linearPredictor=-1.0)
        self.assertAlmostEqual(model.get_risk_for_person(person=None), expit(-1.0))

    def test_scaling_above_one_increases_risk(self):
        model = _StubPrevalenceModel(linearPredictor=-1.0, riskScaling=2.0)
        expected = expit(-1.0 + math.log(2.0))
        self.assertAlmostEqual(model.get_risk_for_person(person=None), expected)
        self.assertGreater(model.get_risk_for_person(person=None), expit(-1.0))

    def test_scaling_below_one_decreases_risk(self):
        model = _StubPrevalenceModel(linearPredictor=-1.0, riskScaling=0.5)
        expected = expit(-1.0 + math.log(0.5))
        self.assertAlmostEqual(model.get_risk_for_person(person=None), expected)
        self.assertLess(model.get_risk_for_person(person=None), expit(-1.0))


class TestEpilepsyPrevalenceDirectMultiply(unittest.TestCase):
    """Epilepsy uses linear_predictor / 1000 (a rate, not a logit), so scaling is applied
    as a direct multiplier rather than an odds shift."""

    def _make_model_with_fixed_lp(self, riskScaling, lp):
        model = EpilepsyPrevalenceModel(riskScaling=riskScaling)
        model.get_linear_predictor_for_person = lambda person: lp
        return model

    def test_default_scaling_is_a_no_op(self):
        model = self._make_model_with_fixed_lp(riskScaling=1.0, lp=2.0)
        self.assertAlmostEqual(model.get_risk_for_person(person=None), 2.0 / 1000.)

    def test_scaling_multiplies_rate_directly(self):
        model = self._make_model_with_fixed_lp(riskScaling=3.0, lp=2.0)
        self.assertAlmostEqual(model.get_risk_for_person(person=None), 3.0 * 2.0 / 1000.)


class TestOutcomePrevalenceModelRepositoryDispatch(unittest.TestCase):
    """The dict passed to OutcomePrevalenceModelRepository must reach the per-outcome
    model instances as `_riskScaling`. Mirror image of the OutcomeModelRepository
    riskScaling dispatch test."""

    def test_default_riskscaling_is_one_for_all_supported_outcomes(self):
        opmr = OutcomePrevalenceModelRepository()
        for outcomeType in (
            OutcomeType.CARDIOVASCULAR,
            OutcomeType.STROKE,
            OutcomeType.DEMENTIA,
            OutcomeType.EPILEPSY,
            OutcomeType.DIABETES,
            OutcomeType.CHRONIC_KIDNEY_DISEASE,
        ):
            inner = opmr._repository[outcomeType]._model
            self.assertEqual(1.0, inner._riskScaling, msg=f"{outcomeType} default")

    def test_dict_propagates_per_outcome(self):
        scaling = {
            OutcomeType.CARDIOVASCULAR: 1.5,
            OutcomeType.STROKE: 2.0,
            OutcomeType.DEMENTIA: 0.5,
            OutcomeType.EPILEPSY: 3.0,
            OutcomeType.DIABETES: 1.25,
            OutcomeType.CHRONIC_KIDNEY_DISEASE: 0.75,
        }
        opmr = OutcomePrevalenceModelRepository(riskScaling=scaling)
        for outcomeType, expected in scaling.items():
            inner = opmr._repository[outcomeType]._model
            self.assertEqual(expected, inner._riskScaling, msg=f"{outcomeType}")

    def test_unspecified_outcomes_default_to_one(self):
        # Only set CV scaling; the others should stay at the 1.0 default.
        opmr = OutcomePrevalenceModelRepository(
            riskScaling={OutcomeType.CARDIOVASCULAR: 4.0}
        )
        self.assertEqual(4.0, opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)
        self.assertEqual(1.0, opmr._repository[OutcomeType.DEMENTIA]._model._riskScaling)
        self.assertEqual(1.0, opmr._repository[OutcomeType.EPILEPSY]._model._riskScaling)


class TestNhanesTrialDescriptionForwardsPrevalenceRiskScaling(unittest.TestCase):
    def test_unset_yields_default_opmr(self):
        desc = NhanesTrialDescription(sampleSize=10, duration=1, year=1999,
                                      personFilters=_adults_filter())
        opmr = desc.peopleArgs["outcomePrevalenceModelRepository"]
        self.assertEqual(1.0, opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)

    def test_dict_reaches_inner_models(self):
        scaling = {OutcomeType.DEMENTIA: 2.5, OutcomeType.EPILEPSY: 1.5}
        desc = NhanesTrialDescription(sampleSize=10, duration=1, year=1999,
                                      personFilters=_adults_filter(),
                                      prevalenceRiskScaling=scaling)
        opmr = desc.peopleArgs["outcomePrevalenceModelRepository"]
        self.assertEqual(2.5, opmr._repository[OutcomeType.DEMENTIA]._model._riskScaling)
        self.assertEqual(1.5, opmr._repository[OutcomeType.EPILEPSY]._model._riskScaling)
        self.assertEqual(1.0, opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)


if __name__ == "__main__":
    unittest.main()
