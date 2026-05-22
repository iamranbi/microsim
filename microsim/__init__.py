"""MICROSIM: a chronic-disease microsimulation framework.

Public API. Import the central entities directly from the package, e.g.::

    from microsim import Person, Population, PopulationFactory

The names below are re-exported here so callers don't depend on the internal
file layout (modules can move without breaking these imports).
"""

from microsim.person.person import Person
from microsim.population.population import Population
from microsim.person.person_factory import PersonFactory
from microsim.population.population_factory import PopulationFactory
from microsim.common.population_type import PopulationType

__all__ = [
    "Person",
    "Population",
    "PersonFactory",
    "PopulationFactory",
    "PopulationType",
]
