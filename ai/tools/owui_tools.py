"""OWUI-format source code for the 4 OTAI pedagogical tools.

Each constant is a Python source string stored in the DB.
The frontend reads it, extracts the OpenAI function-calling schema,
and executes tool calls server-side.
"""

GENERATE_CHART_CODE = '''
class Tools:
    """Visualisation de fonctions et de données sous forme de graphes ASCII."""

    def plot_function(self, expression: str) -> str:
        """
        Trace le graphe d\'une fonction mathématique (ASCII).
        Utilise cet outil quand l\'utilisateur demande de visualiser
        y = f(x), une parabole, une courbe ou une fonction.
        :param expression: expression mathématique, ex: "x**2 - 8"
        :return: graphe ASCII
        """
        from ai.tools.generate_chart import generate_chart
        return generate_chart.invoke(f"function {expression}")

    def plot_timeline(self, description: str) -> str:
        """
        Génère une timeline ou un diagramme de Gantt ASCII.
        :param description: description des étapes à afficher
        :return: timeline ASCII
        """
        from ai.tools.generate_chart import generate_chart
        return generate_chart.invoke(f"timeline {description}")

    def plot_bar_chart(self, description: str) -> str:
        """
        Génère un diagramme en barres ASCII.
        :param description: données à représenter
        :return: bar chart ASCII
        """
        from ai.tools.generate_chart import generate_chart
        return generate_chart.invoke(f"bar {description}")
'''

MATH_EVALUATOR_CODE = '''
class Tools:
    """Calcul symbolique et numérique d\'expressions mathématiques."""

    def evaluate_expression(self, expression: str) -> str:
        """
        Évalue une expression mathématique (algèbre, calcul, simplification).
        Utilise cet outil pour résoudre des équations, calculer des dérivées,
        simplifier des expressions ou obtenir une valeur numérique.
        :param expression: expression mathématique, ex: "x**2 + 2*x - 8"
        :return: résultat numérique ou symbolique formaté en markdown
        """
        from ai.tools.math_evaluator import math_evaluator
        return math_evaluator.invoke(expression)
'''

LIVE_CODE_CODE = '''
class Tools:
    """Exécution de code Python dans un environnement isolé."""

    def run_python(self, code: str) -> str:
        """
        Exécute du code Python et retourne la sortie (stdout + stderr).
        Utilise cet outil pour tester du code, valider une solution
        ou démontrer un comportement d\'exécution.
        Timeout : 5 secondes. Pas d\'accès réseau ni système de fichiers.
        :param code: code Python à exécuter
        :return: sortie d\'exécution (stdout/stderr)
        """
        from ai.tools.live_code_evaluation import live_code_evaluation
        return live_code_evaluation.invoke(code)
'''

SEARCH_WEB_CODE = '''
class Tools:
    """Recherche web pour enrichir les réponses pédagogiques."""

    def search(self, query: str) -> str:
        """
        Recherche des informations sur le web (DuckDuckGo).
        Utilise cet outil quand la réponse nécessite des faits récents,
        de la documentation officielle ou des exemples du monde réel.
        :param query: requête de recherche en langage naturel
        :return: résumé des résultats web
        """
        from ai.tools.search_web import search_web
        return search_web.invoke(query)
'''

OTAI_TOOLS = [
    {
        "id": "otai_generate_chart",
        "name": "OpenTutorAI — Graphes & Visualisation",
        "content": GENERATE_CHART_CODE,
        "meta": {
            "description": "Trace des fonctions mathématiques et des diagrammes ASCII"
        },
    },
    {
        "id": "otai_math_evaluator",
        "name": "OpenTutorAI — Calcul Symbolique",
        "content": MATH_EVALUATOR_CODE,
        "meta": {"description": "Évalue des expressions mathématiques via sympy"},
    },
    {
        "id": "otai_live_code",
        "name": "OpenTutorAI — Exécution de Code",
        "content": LIVE_CODE_CODE,
        "meta": {
            "description": "Exécute du code Python dans un subprocess isolé (5s timeout)"
        },
    },
    {
        "id": "otai_search_web",
        "name": "OpenTutorAI — Recherche Web",
        "content": SEARCH_WEB_CODE,
        "meta": {
            "description": "Recherche DuckDuckGo pour enrichir les réponses pédagogiques"
        },
    },
]
