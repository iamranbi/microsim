"""The Population, its factory, and the population-level repositories.

    from microsim.population import Population, PopulationFactory

``population.py`` lives here as ``microsim.population.population``; ``Population`` is
re-exported so ``from microsim.population import Population`` keeps working. Internal
modules import the concrete path (``from microsim.population.population import
Population``) to avoid import cycles.
"""

from microsim.population.population import Population
from microsim.population.population_factory import PopulationFactory
from microsim.population.population_model_repository import (
    PopulationModelRepository,
    PopulationRepositoryType,
)
from microsim.population.standardized_population import StandardizedPopulation
from microsim.population.initialization_repository import InitializationRepository

__all__ = [
    "Population",
    "PopulationFactory",
    "PopulationModelRepository",
    "PopulationRepositoryType",
    "StandardizedPopulation",
    "InitializationRepository",
]
