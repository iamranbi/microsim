from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.cv_model_repository import CVPrevalenceModelRepository
from microsim.outcomes.stroke_partition_model_repository import StrokePrevalenceModelRepository
from microsim.outcomes.mi_partition_model_repository import MIPrevalenceModelRepository
from microsim.outcomes.dementia_model_repository import DementiaPrevalenceModelRepository
from microsim.outcomes.diabetes_model_repository import DiabetesPrevalenceModelRepository
from microsim.outcomes.chronic_kidney_disease_model_repository import (
    ChronicKidneyDiseasePrevalenceModelRepository,
)
from microsim.outcomes.epilepsy_model_repository import EpilepsyPrevalenceModelRepository
from microsim.outcomes.cognition_model_repository import CognitionPrevalenceModelRepository


class OutcomePrevalenceModelRepository:
    """Holds the priorToSim (prevalence) rules for outcomes.
       Mirror image of OutcomeModelRepository: every OutcomeType is registered as a key, but
       outcomes without a prevalence model resolve to None and are skipped during seeding.

       riskScaling: optional dict[OutcomeType, float] applied per-outcome inside each
       prevalence model. For expit-based logistic models the scalar is applied as an odds
       shift (lp + log(scaling)); for the epilepsy rate model it is a direct rate multiplier.
       Outcomes without a probabilistic prevalence model (MI partition, cognition GCP) ignore
       riskScaling regardless of what is passed."""

    def __init__(self, riskScaling=None):
        if riskScaling is None:
            riskScaling = {}
        cvScaling = riskScaling.get(OutcomeType.CARDIOVASCULAR, 1.0)
        strokeScaling = riskScaling.get(OutcomeType.STROKE, 1.0)
        dementiaScaling = riskScaling.get(OutcomeType.DEMENTIA, 1.0)
        epilepsyScaling = riskScaling.get(OutcomeType.EPILEPSY, 1.0)
        diabetesScaling = riskScaling.get(OutcomeType.DIABETES, 1.0)
        ckdScaling = riskScaling.get(OutcomeType.CHRONIC_KIDNEY_DISEASE, 1.0)
        self._repository = {
            OutcomeType.WMH:                      None,
            OutcomeType.COGNITION:                CognitionPrevalenceModelRepository(),
            OutcomeType.CI:                       None,
            OutcomeType.MCI:                      None,
            OutcomeType.DIABETES:                 DiabetesPrevalenceModelRepository(riskScaling=diabetesScaling),
            OutcomeType.CHRONIC_KIDNEY_DISEASE:   ChronicKidneyDiseasePrevalenceModelRepository(riskScaling=ckdScaling),
            OutcomeType.CARDIOVASCULAR:           CVPrevalenceModelRepository(riskScaling=cvScaling),
            OutcomeType.STROKE:                   StrokePrevalenceModelRepository(riskScaling=strokeScaling),
            OutcomeType.MI:                       MIPrevalenceModelRepository(),
            OutcomeType.NONCARDIOVASCULAR:        None,
            OutcomeType.DEMENTIA:                 DementiaPrevalenceModelRepository(riskScaling=dementiaScaling),
            OutcomeType.EPILEPSY:                 EpilepsyPrevalenceModelRepository(riskScaling=epilepsyScaling),
            OutcomeType.DEATH:                    None,
            OutcomeType.QUALITYADJUSTED_LIFE_YEARS: None,
        }
        self.check_repository_completeness()

    def check_repository_completeness(self):
        for outcome in OutcomeType:
            if outcome not in list(self._repository.keys()):
                raise RuntimeError("OutcomePrevalenceModelRepository is incomplete")

    def has_prevalence_model(self, outcomeType):
        return self._repository.get(outcomeType) is not None
