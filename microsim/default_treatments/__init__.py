"""Default ("usual care") treatments: the core default-treatment enumerations.

    from microsim.default_treatments import DefaultTreatmentsType

(``DefaultTreatmentModelRepository`` is left on its module to keep ``import
microsim`` light — import it from
``microsim.default_treatments.default_treatment_model_repository``.)
"""

from microsim.default_treatments.default_treatments import (
    DefaultTreatmentsType,
    CategoricalDefaultTreatmentsType,
    ContinuousDefaultTreatmentsType,
)

__all__ = [
    "DefaultTreatmentsType",
    "CategoricalDefaultTreatmentsType",
    "ContinuousDefaultTreatmentsType",
]
