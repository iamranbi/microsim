"""The Person agent, its factory, and person filters.

    from microsim.person import Person, PersonFactory

``person.py`` lives here as ``microsim.person.person``; ``Person`` is re-exported so
``from microsim.person import Person`` keeps working. Internal modules import the
concrete path (``from microsim.person.person import Person``) to avoid import cycles.
"""

from microsim.person.person import Person
from microsim.person.person_factory import PersonFactory
from microsim.person.person_filter import PersonFilter
from microsim.person.person_filter_factory import PersonFilterFactory

__all__ = ["Person", "PersonFactory", "PersonFilter", "PersonFilterFactory"]
