"""Configuration pytest pour les tests externes (LongMemEval-S, MRBench).

Ces tests font des appels LLM réels et écrivent dans des fichiers SQLite temporaires.
Ils DOIVENT s'exécuter séquentiellement — ne jamais les lancer avec pytest-xdist (-n auto).
"""

import pytest


def pytest_collection_modifyitems(items):
    """Force l'exécution séquentielle de tous les tests du package external."""
    external_items = [i for i in items if "eval/external" in str(i.fspath)]
    for item in external_items:
        # Désactive xdist pour ce test si le plugin est actif
        item.add_marker(pytest.mark.xdist_group("external_serial"))
