"""Shared value types used across the person and population layers.

    from microsim.common import AgeScope, VariableType, PopulationType

(The ``data_loader`` I/O helpers live on ``microsim.common.data_loader``; they're
not re-exported here so importing the value types stays dependency-free.)
"""

from microsim.common.age_scope import AgeScope
from microsim.common.variable_type import VariableType
from microsim.common.population_type import PopulationType

__all__ = ["AgeScope", "VariableType", "PopulationType"]
