# Scénarios d'évaluation — Open TutorAI

---

## Problématique et objectifs de l'évaluation

Open TutorAI repose sur une architecture hybride combinant mémoire persistante, moteur de récupération contextuelle (RAG), compression de contexte et orchestration agentique LangGraph. La question centrale de l'évaluation est la suivante :

> **Dans quelle mesure cette architecture permet-elle d'assurer un suivi pédagogique personnalisé, cohérent et robuste sur la durée ?**

Les métriques retenues répondent chacune à une dimension de cette question :

| Dimension | Métriques associées | Scénarios |
|-----------|-------------------|-----------|
| Récupération documentaire | Recall@k, Faithfulness, Learning Gain, Latence | S0 |
| Fidélité de la mémoire | MRA | S1, S3, S5 |
| Efficacité de la compression | CR, IRR | S2, S6 |
| Robustesse de l'orchestration | LRA | S4 |
| Progression pédagogique | Learning Gain, TCR | S5 |
| Adaptation au profil | Note humaine | S7 |
| Robustesse aux entrées inattendues | Absence de crash, note humaine | S8 |

---

## Environnement d'exécution

Toutes les exécutions doivent être réalisées dans l'environnement suivant pour garantir la reproductibilité :

| Paramètre | Valeur |
|-----------|--------|
| Modèle LLM | À fixer avant l'exécution (ex : `llama3.2`, `gpt-4o-mini`) |
| Température LLM | `0` (déterministe) |
| Version de l'API | `http://localhost:8080/api/v1` |
| Base de données | SQLite — fichier réinitialisé avant chaque scénario |
| Tokenizer | `tiktoken` (cl100k_base) — version fixée |
| Environnement Python | `~/.pyenv/versions/tutorai-env` |
| Système d'exploitation | Linux — éviter l'exécution sur des environnements différents entre deux runs |

---

## Ordre d'exécution et dépendances

Certains scénarios sont indépendants, d'autres supposent qu'un état a été créé par un scénario précédent. Le tableau suivant rend ces dépendances explicites :

| Scénario | Dépend de | Raison |
|----------|-----------|--------|
| S0 | — | Indépendant |
| S1 | — | Indépendant |
| S2 | — | Indépendant |
| S3 | S1 | Suppose qu'un profil « débutant » existe en base |
| S4 | — | Indépendant |
| S5 | — | Indépendant (crée son propre historique sur 4 sessions) |
| S6 | S2 | Suppose qu'un contexte compressé existe pour tester la résistance |
| S7 | — | Indépendant |
| S8 | — | Indépendant |

**Ordre recommandé :** S0 → S1 → S3 → S2 → S6 → S4 → S5 → S7 → S8

Chaque scénario démarre avec un reset complet de la base et des mémoires, **sauf S3 et S6** qui héritent de l'état produit par leur dépendance.

---

## Justification des seuils

Les seuils ne sont pas arbitraires. Chacun est justifié par une contrainte fonctionnelle ou une référence du domaine :

| Métrique | Seuil | Justification |
|----------|-------|---------------|
| Recall@5 | ≥ 0,80 | En dessous de 80 %, les documents les plus pertinents ne figurent pas dans les résultats transmis au LLM — la réponse risque d'être incomplète ou erronée dès la première interaction. |
| Faithfulness | ≥ 0,90 | Un taux de 90 % signifie au plus 1 affirmation sur 10 non ancrée dans les sources — seuil correspondant aux benchmarks RAGAs pour un usage pédagogique où toute hallucination peut induire l'apprenant en erreur. |
| Latence | ≤ 5 s | Au-delà de 5 s, l'expérience pédagogique est perturbée (seuil issu des recherches sur l'attention en e-learning, Nielsen 1993 actualisé). |
| MRA | 100 % | La mémoire d'un tuteur est binaire : une information partiellement restituée produit un suivi incohérent. Aucune perte n'est acceptable. |
| CR | ≤ 0,30 | Une fenêtre de contexte typique est de 4 096 tokens. Pour absorber un historique de 10 000 tokens, une réduction d'au moins 70 % est nécessaire. |
| IRR | ≥ 0,80 | En dessous de 80 % de rétention, les concepts clés et difficultés détectées risquent d'être perdus, rendant le résumé inutilisable pour le suivi pédagogique. |
| LRA | ≥ 80 % | Un routeur aléatoire entre 6 agents obtient ~17 %. Le seuil de 80 % garantit que le LLM apporte une valeur réelle par rapport au fallback déterministe, tout en tolérant 20 % de cas limites. |
| TCR | ≥ 95 % | Une tâche pédagogique non complétée laisse l'apprenant sans réponse. Le 5 % de tolérance couvre les timeouts réseau et cas extrêmes. |
| Learning Gain | ≥ 0,30 | Seuil de « gain moyen » issu des travaux de Hake (1998) en éducation, largement adopté pour évaluer l'efficacité d'un dispositif pédagogique. |
| Note humaine | ≥ 4/5 | Correspond à « bon » sur une échelle de Likert standard. En dessous de 4, la réponse est jugée insuffisante par l'évaluateur. |

---

## Vue d'ensemble des scénarios

| # | Scénario | Thème | Évaluation | Métrique principale |
|---|----------|-------|-----------|-------------------|
| S0 | Acquisition d'une notion | RAG + mémoire initiale | Mixte | Recall@5 ≥ 0,80 + Faithfulness ≥ 0,90 |
| S1 | Vérification de la mémoire | Mémoire inter-sessions | Automatisée | MRA = 100 % |
| S2 | Compression de contexte | Contexte intra-session | Automatisée | CR ≤ 0,30 |
| S3 | Conflit mémoire | Versionnement profil | Automatisée | MRA + cohérence historique |
| S4 | Routage LLM-first | Orchestration agentique | Automatisée | LRA ≥ 80 % |
| S5 | Apprentissage longitudinal | Mémoire long terme | Mixte | MRA + TCR + Learning Gain ≥ 0,30 |
| S6 | Résistance à la perte de contexte | Contexte + compression | Mixte | CR ≤ 0,30 + IRR ≥ 0,80 + note ≥ 4/5 |
| S7 | Adaptation pédagogique | Adaptation | Humaine | Note qualitative ≥ 4/5 |
| S8 | Entrée hors-sujet ou ambiguë | Robustesse | Mixte | 0 crash + note redirection ≥ 4/5 |

---

## Scénarios automatisés

### S0 — Acquisition d'une notion

**Objectif :** vérifier que le système récupère les documents pédagogiques pertinents, produit une explication adaptée au niveau débutant, enregistre ce niveau en mémoire et répond dans un délai acceptable.

**Déroulement :**
1. **Pré-test** : quiz standardisé de 5 questions sur les boucles `for` en Python (score initial enregistré).
2. Un apprenant débutant soumet la question : « Peux-tu m'expliquer les boucles `for` en Python ? »
3. Le pipeline RAG récupère les documents pédagogiques du corpus versionné.
4. Le système génère une explication adaptée à un niveau débutant, ancrée dans les documents récupérés.
5. Le niveau « débutant » est enregistré en mémoire pour la session.
6. **Post-test** : même quiz de 5 questions — calcul du Learning Gain à partir des scores pré/post.
7. Vérification que les documents pertinents figurent dans le top 5, que la réponse n'introduit aucune affirmation non documentée, et que la latence reste sous le seuil.

**Métriques :**
| Métrique | Cible | Évaluation |
|----------|-------|-----------|
| Recall@5 | ≥ 0,80 | Automatisée |
| Faithfulness | ≥ 0,90 | Automatisée (LLM juge) |
| Learning Gain | ≥ 0,30 | Humaine (double évaluation) |
| Latence end-to-end | ≤ 5 s | Automatisée |

**Définitions :**

> **Recall@k**
>
> `Recall@k = |D_pertinents ∩ D_top_k| / |D_pertinents|`
>
> - `D_pertinents` : ensemble des documents jugés pertinents pour la question, établi manuellement avant le test (ground truth)
> - `D_top_k` : ensemble des k documents renvoyés par le pipeline de récupération
> - k = 5 dans ce scénario
>
> Mesure la capacité du pipeline RAG à retrouver les bonnes sources. Un Recall@5 = 0,80 signifie que 4 des 5 documents pertinents figurent dans les 5 premiers résultats.

> **Faithfulness**
>
> `Faithfulness = A_ancrées / A_total`
>
> - `A_total` : nombre total d'affirmations factuelles dans la réponse générée
> - `A_ancrées` : nombre de ces affirmations dont la source est traçable dans les documents récupérés
>
> Évaluée par un LLM juge (température 0) comparant chaque affirmation aux extraits sources. Un score < 0,90 signale une hallucination partielle incompatible avec un usage pédagogique fiable.

> **Learning Gain (LG)** — voir S5.

> **Latence end-to-end**
>
> Temps écoulé entre la réception de la requête HTTP et l'envoi complet de la réponse, mesuré côté client. Inclut récupération RAG, génération LLM et écriture mémoire.

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Fixture d'état initial | Reset complet BD + mémoires avant l'exécution |
| Corpus documentaire versionné | 10 documents pédagogiques sur les boucles Python — mêmes documents à chaque run |
| Ground truth RAG fixée | Liste des 3 documents pertinents établie manuellement avant le test, non modifiable après exécution |
| Pré-test standardisé | Quiz fixe de 5 questions administré avant la session |
| Double évaluation humaine (LG) | Deux évaluateurs indépendants — accord ≥ 80 % requis, arbitrage par un tiers sinon |
| Données de test distinctes | Domaine Python/boucles — corpus différent des données de développement |

**Tests correspondants :** `tests/eval/test_s0_notion_acquisition.py`
| Test | Vérifie |
|------|---------|
| `test_s0_recall_at_5_above_target` | Recall@5 ≥ 0,80 sur le corpus de 10 documents versionnés |
| `test_s0_faithfulness_above_target` | Faithfulness ≥ 0,90 : toutes les affirmations sont ancrées dans les sources |
| `test_s0_level_stored_in_memory` | Niveau « débutant » correctement persisté en base après la session |
| `test_s0_response_adapted_to_beginner` | La réponse ne contient pas de termes avancés non expliqués |
| `test_s0_latency_within_budget` | Latence end-to-end ≤ 5 s mesurée côté client |
| `test_s0_learning_gain_framework` | Calcule et documente le LG pour double évaluation humaine |

---

### S1 — Vérification de la mémoire

**Objectif :** vérifier que le système restitue fidèlement les informations d'une session précédente dans une nouvelle session.

**Déroulement :**
1. Session 1 : l'apprenant étudie les balises HTML à un niveau débutant.
2. Session 2 (nouvelle session) : l'apprenant demande de poursuivre le cours précédent.
3. Le système doit se souvenir que le sujet était HTML, que le concept étudié était les balises et que le niveau était débutant.
4. Vérification du contenu exact de chaque champ récupéré (valeur, pas seulement présence).

**Métriques :**
| Métrique | Cible |
|----------|-------|
| Memory Retrieval Accuracy | 100 % |

**Définition :**

> **Memory Retrieval Accuracy (MRA)**
>
> `MRA = M_correctes / M_attendues`
>
> - `M_correctes` : nombre d'informations correctement récupérées (valeur exacte vérifiée)
> - `M_attendues` : nombre total d'informations attendues
>
> Une valeur de 100 % indique que le profil apprenant et l'historique pédagogique sont restitués sans perte d'information.

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Fixture d'état initial | Reset complet BD + mémoires avant l'exécution |
| Données de test distinctes | Domaine HTML — différent des données de développement (Python) |
| Assertion sur le contenu | Vérification de la valeur exacte de chaque champ (`sujet = "HTML"`, `niveau = "débutant"`), pas seulement leur présence |

**Tests correspondants :** `tests/eval/test_s1_memory_retrieval.py`
| Test | Vérifie |
|------|---------|
| `test_s1_memory_stored_and_retrieved` | MRA = 100 % sur les 4 champs attendus |
| `test_s1_mra_field_by_field` | Assertion champ par champ pour localiser précisément toute perte |
| `test_s1_new_session_isolation` | Pas de fuite de mémoire entre deux utilisateurs distincts |
| `test_s1_no_memory_returns_empty` | Aucune hallucination quand aucune session précédente n'existe |

---

### S2 — Compression de contexte

**Objectif :** valider que la compression agressive du contexte ne sacrifie pas l'information critique.

**Déroulement :**
1. Un corpus de conversation fixe et reproductible de 10 000 tokens est injecté (même fichier à chaque exécution).
2. Le pipeline de résumé est déclenché.
3. Le contexte est réduit à 2 500 tokens (Compression Ratio = 0,25).
4. Vérification que le ratio est inférieur ou égal à la cible de 0,30.

**Métriques :**
| Métrique | Cible |
|----------|-------|
| Compression Ratio | ≤ 0,30 |

**Définition :**

> **Compression Ratio (CR)**
>
> `CR = Tokens_après / Tokens_avant`
>
> - `Tokens_avant` : nombre de tokens avant compression
> - `Tokens_après` : nombre de tokens après compression
>
> Plus le ratio est faible, plus la compression est agressive. Le seuil de 0,30 garantit une réduction d'au moins 70 % du contexte.

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Fixture d'état initial | Reset complet BD + mémoires avant l'exécution |
| Input fixe et reproductible | Corpus de conversation tokenisé et versionné — même fichier à chaque run |
| Comptage de tokens déterministe | Tokenizer `tiktoken cl100k_base` — version fixée |

**Tests correspondants :** `tests/eval/test_s2_compression.py`
| Test | Vérifie |
|------|---------|
| `test_s2_compression_ratio_within_target` | CR ≤ 0,30 sur un corpus de ~10 000 tokens |
| `test_s2_summary_is_non_empty` | Le résumé produit n'est pas vide |
| `test_s2_summary_cached_after_compression` | Le résumé est bien persisté en base après compression |
| `test_s2_compression_reproducible` | Deux exécutions avec le même corpus produisent le même CR |

---

### S3 — Conflit mémoire

**Objectif :** vérifier que le système détecte une contradiction de profil, met à jour le niveau et conserve l'historique versionné.

**Déroulement :**
1. Session 1 : l'apprenant se déclare débutant en SQL. *(hérite de l'état créé par S1)*
2. Sessions 2 et 3 : interactions normales au niveau débutant.
3. Session 4 : l'apprenant indique pratiquer SQL depuis deux ans.
4. Le système doit détecter la contradiction avec le profil enregistré.
5. Mise à jour du niveau : débutant → avancé, adaptation immédiate du contenu.
6. L'ancien profil n'est pas écrasé mais versionné avec horodatage.
7. Vérification directe en base : champ `niveau` = « avancé », « débutant » conservé dans l'historique horodaté.

**Métriques :**
| Métrique | Cible |
|----------|-------|
| Memory Retrieval Accuracy | 100 % |
| Cohérence historique versionné | Présence des deux entrées horodatées avec valeurs exactes |

**Définitions :**

> **Memory Retrieval Accuracy (MRA)** — voir S1.
>
> Vérifie ici que le niveau mis à jour (« avancé ») est bien restitué et que l'entrée historique (« débutant ») est conservée avec son horodatage.

> **Cohérence historique versionné**
>
> Vérification structurelle : interrogation directe de la base après la session 4.
> - Champ `niveau` = « avancé » (valeur courante, vérifiée par égalité de chaîne)
> - Entrée « débutant » présente dans l'historique avec horodatage antérieur à la session 4

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Dépendance | Exécuter S1 avant S3 — S3 hérite du profil créé en S1 |
| Données de test distinctes | Domaine SQL — différent des données de développement (Python) |
| Assertion sur le contenu | Vérification de la valeur exacte et de l'horodatage |

**Tests correspondants :** `tests/eval/test_s3_memory_conflict.py`
| Test | Vérifie |
|------|---------|
| `test_s3_current_profile_updated_to_avance` | Niveau courant = « avancé » après le conflit |
| `test_s3_historical_profile_preserved` | Entrée « débutant » conservée (non écrasée) |
| `test_s3_history_is_ordered_chronologically` | Horodatage « débutant » antérieur à « avancé » |
| `test_s3_mra_after_conflict` | MRA = 100 % sur les 3 champs attendus (niveau courant, historique, horodatage) |
| `test_s3_no_cross_user_contamination` | Le conflit d'un utilisateur ne modifie pas le profil d'un autre |

---

### S4 — Routage LLM-first

**Objectif :** valider que `_llm_route` prend les bonnes décisions de routage en conditions normales et bascule correctement sur le fallback déterministe en cas d'échec.

**Déroulement :**
1. Un ensemble fixe de 10 états courants versionnés est injecté séquentiellement.
2. Pour chaque état, l'orchestrateur appelle `_llm_route` avec : sujet, niveau, concepts faibles, trace des agents, nombre de tentatives, résultat de vérification.
3. Le LLM retourne en JSON l'agent suivant + raisonnement + indice de confiance.
4. Vérification des quatre points :
   - Réponse JSON valide → décision tracée `[LLM]`, `agent_reasoning` peuplé, indice de confiance consigné.
   - Réponse invalide ou agent inconnu → repli `_route()` tracé `[fallback]`.
5. Mesure de la proportion de décisions prises sans recours au fallback.

**Métriques :**
| Métrique | Cible |
|----------|-------|
| LLM Routing Accuracy | ≥ 80 % |

**Définition :**

> **LLM Routing Accuracy (LRA)**
>
> `LRA = R_LLM / R_total`
>
> - `R_LLM` : nombre de décisions prises par `_llm_route` avec une réponse JSON valide
> - `R_total` : nombre total de décisions de routage observées
>
> Baseline de référence : un routeur aléatoire entre 6 agents obtient ~17 %. Le seuil de 80 % garantit une valeur ajoutée réelle du LLM par rapport au fallback déterministe.

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Fixture d'état initial | Reset complet BD + mémoires avant l'exécution |
| Température LLM fixée à 0 | Élimine le non-déterminisme — même input → même output |
| Inputs de routage versionnés | 10 états courants fixes (fichier JSON versionné) — reproductibles à l'identique |

**Tests correspondants :** `tests/eval/test_s4_routing.py`
| Test | Vérifie |
|------|---------|
| `test_s4_lra_above_target` | LRA = R_LLM / R_total ≥ 0,80 sur 10 états versionnés |
| `test_s4_fallback_activates_on_invalid_llm_response` | `_route()` prend le relais quand le LLM retourne None |
| `test_s4_llm_invalid_json_triggers_fallback` | JSON malformé → fallback, sans exception |
| `test_s4_llm_unknown_agent_triggers_fallback` | Agent inconnu dans la réponse → fallback |
| `test_s4_orchestrator_node_traces_llm_decision` | `[LLM]` tracé dans `agent_trace` quand le LLM répond |
| `test_s4_orchestrator_node_traces_fallback` | `[fallback]` tracé dans `agent_trace` quand le LLM échoue |
| `test_s4_deterministic_route_order` | Ordre exact : memory → knowledge → diagnostics → planner → exercise → verifier |
| `test_s4_planner_retry_on_needs_review` | Retour vers planner si `needs_review` et retries < 3 |
| `test_s4_planner_retry_stops_at_max` | Plus de retry vers planner après 3 tentatives → feedback |

---

## Scénarios mixtes (métriques + validation humaine)

### S5 — Apprentissage longitudinal

**Objectif :** scénario le plus représentatif de la mémoire à long terme — vérifier que le système cumule et exploite correctement les informations sur plusieurs sessions.

**Déroulement :**
1. **Pré-test** : quiz standardisé de 10 questions sur les fonctions JavaScript (score initial enregistré).
2. Session 1 — Apprentissage initial : introduction aux fonctions JavaScript, enregistrement du niveau et des notions vues.
3. Session 2 — Difficulté : l'apprenant signale des difficultés sur les paramètres, enregistrement de la difficulté détectée.
4. Session 3 — Quiz intermédiaire : exercices sur les fonctions, mise à jour de la probabilité de maîtrise (BKT).
5. Session 4 — Montée en niveau : le système détecte la progression et adapte le contenu.
6. **Post-test** : même quiz standardisé de 10 questions — calcul du Learning Gain à partir des scores pré/post.
7. Vérification que le système récupère correctement les notions étudiées, les difficultés détectées et le niveau de maîtrise estimé.

**Métriques :**
| Métrique | Cible | Évaluation |
|----------|-------|-----------|
| Memory Retrieval Accuracy | 100 % | Automatisée |
| Task Completion Rate | ≥ 95 % | Automatisée |
| Learning Gain | ≥ 0,30 | Humaine (double évaluation) |

**Définitions :**

> **Task Completion Rate (TCR)**
>
> `TCR = N_succès / N_total`
>
> - `N_succès` : nombre de sessions complétées sans blocage
> - `N_total` : nombre total de sessions exécutées

> **Learning Gain (LG)**
>
> `LG = (score_post − score_pré) / (score_max − score_pré)`
>
> Seuil de 0,30 issu des travaux de Hake (1998) — correspond à un gain pédagogique « moyen », considéré comme le minimum acceptable pour un dispositif d'enseignement efficace.

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Fixture d'état initial | Reset complet BD + mémoires avant l'exécution |
| Données de test distinctes | Domaine JavaScript — différent des données de développement (Python) |
| Pré-test standardisé | Quiz fixe de 10 questions administré avant la session 1 |
| Double évaluation humaine | Deux évaluateurs indépendants — accord ≥ 80 % requis, arbitrage par un tiers sinon |

**Tests correspondants :** `tests/eval/test_s5_longitudinal.py`
| Test | Vérifie |
|------|---------|
| `test_s5_mra_across_4_sessions` | MRA = 100 % : chaque session a ses champs essentiels peuplés |
| `test_s5_tcr_above_target` | TCR = N_succès / N_total ≥ 0,95 |
| `test_s5_level_progression_detected` | Montée en niveau détectée entre session 2 et session 3 |
| `test_s5_difficulties_recorded` | Difficultés signalées en session 2 correctement persistées |
| `test_s5_notions_cumulated_across_sessions` | Notions vues s'accumulent sur les 4 sessions (≥ 6 distinctes) |
| `test_s5_learning_gain_framework` | Calcule et documente le LG pour double évaluation humaine |

---

### S6 — Résistance à la perte de contexte

**Objectif :** vérifier que la compression automatique déclenchée par dépassement de la fenêtre de contexte préserve les informations pédagogiques essentielles.

**Déroulement :**
1. Le corpus compressé produit par S2 est rechargé. *(dépendance sur S2)*
2. Un corpus de session complémentaire est injecté pour atteindre 12 000 tokens au total.
3. La compression automatique est déclenchée par dépassement de la fenêtre.
4. L'apprenant pose la question : « rappelle-moi ce qu'on a vu sur les listes ».
5. Vérification du Compression Ratio, de l'Information Retention Rate et de la cohérence de la réponse post-compression.

**Métriques :**
| Métrique | Cible | Évaluation |
|----------|-------|-----------|
| Compression Ratio | ≤ 0,30 | Automatisée |
| Information Retention Rate | ≥ 0,80 | Automatisée |
| Cohérence réponse post-compression | ≥ 4/5 | Humaine (double évaluation) |

**Définitions :**

> **Compression Ratio (CR)** — voir S2.

> **Information Retention Rate (IRR)**
>
> `IRR = I_conservées / I_clés`
>
> - `I_clés` : liste des informations pédagogiques jugées essentielles dans le corpus original (concepts vus, difficultés détectées, niveau estimé) — établie manuellement avant le test
> - `I_conservées` : nombre de ces informations présentes dans le résumé compressé
>
> L'IRR est le contrepoids du CR : il garantit que la réduction du contexte ne sacrifie pas les éléments pédagogiques essentiels. CR et IRR doivent être lus conjointement — un CR de 0,10 qui dégrade l'IRR sous 0,80 est un échec.

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Dépendance | Exécuter S2 avant S6 |
| Input fixe et reproductible | Corpus de session versionné — même fichier à chaque run |
| Liste I_clés établie a priori | La liste des informations essentielles est fixée avant l'exécution, pas après |
| Double évaluation humaine | Deux évaluateurs indépendants — accord ≥ 80 % requis |

**Tests correspondants :** `tests/eval/test_s6_context_loss.py`
| Test | Vérifie |
|------|---------|
| `test_s6_compression_ratio_within_target` | CR ≤ 0,30 sur un corpus de ~12 000 tokens |
| `test_s6_irr_above_target` | IRR ≥ 0,80 : les 6 informations clés sont conservées dans le résumé |
| `test_s6_cr_and_irr_not_both_degraded` | Trade-off : si CR < 0,10, l'IRR ne doit pas chuter sous 0,80 |
| `test_s6_degraded_summary_irr_below_target` | Valide la sensibilité de l'IRR — un résumé vague doit scorer < 0,80 |
| `test_s6_post_compression_response_framework` | Documente la réponse post-compression pour double évaluation humaine |

---

## Scénarios à évaluation humaine

### S7 — Adaptation pédagogique

**Objectif :** vérifier que le système détecte un signal de difficulté explicite et adapte qualitativement son explication.

**Déroulement :**
1. L'apprenant signale que les requêtes SQL restent difficiles malgré une première explication.
2. Le système doit détecter la difficulté, diminuer la complexité de l'explication et proposer un nouvel exemple.
3. Une réponse équivalente est générée pour un apprenant avancé sur le même concept.
4. Deux évaluateurs humains indépendants comparent les deux réponses à l'aide d'une grille standardisée.

**Métriques :**
| Métrique | Cible | Évaluation |
|----------|-------|-----------|
| Pertinence de la simplification | Confirmée | Humaine (double évaluation) |
| Note qualitative | ≥ 4/5 | Humaine (double évaluation) |

**Grille de notation standardisée :**
| Critère | Poids |
|---------|-------|
| Lisibilité de l'explication simplifiée | 25 % |
| Pertinence de l'exemple proposé | 25 % |
| Absence de jargon non expliqué | 25 % |
| Différence perceptible avec la réponse avancée | 25 % |

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Données de test distinctes | Domaine SQL — différent des données de développement (Python) |
| Double évaluation humaine | Deux évaluateurs indépendants — accord ≥ 80 % requis, arbitrage par un tiers sinon |
| Grille de notation standardisée | Critères fixes soumis aux deux évaluateurs avant la lecture des réponses |

**Tests correspondants :** `tests/eval/test_s7_adaptation.py`
| Test | Vérifie |
|------|---------|
| `test_s7_two_responses_produced` | Le système produit bien deux réponses non vides |
| `test_s7_responses_are_different` | Les deux réponses ne sont pas identiques |
| `test_s7_beginner_response_less_technical` | La réponse débutant contient moins de termes techniques |
| `test_s7_beginner_response_contains_concrete_example` | La réponse débutant inclut un exemple concret |
| `test_s7_no_unexplained_jargon_in_beginner_response` | Pas de jargon SQL avancé non expliqué pour le débutant |
| `test_s7_human_evaluation_framework` | Documente les deux réponses et la grille pour double évaluation humaine |

---

## Scénario proposé (hors Chapitre 7)

### S8 — Entrée hors-sujet ou ambiguë

**Objectif :** valider la robustesse du système face à des entrées inattendues, sans plantage ni réponse incohérente.

**Déroulement :**
1. En cours de session pédagogique, l'apprenant soumet une entrée issue d'un corpus standardisé de 4 types.
2. Le système ne doit pas générer d'exception non gérée.
3. Il ne doit pas produire de réponse hors-cadre pédagogique.
4. Il doit rediriger l'apprenant vers le sujet en cours ou demander une clarification.
5. Vérification que la mémoire et l'état de session ne sont pas corrompus après l'entrée.

**Cas de test standardisés :**
| # | Type | Exemple d'entrée |
|---|------|-----------------|
| 1 | Hors-domaine | « Comment faire une quiche lorraine ? » |
| 2 | Ambiguë | « Continue » (sans contexte actif) |
| 3 | Vide | `""` (chaîne vide) |
| 4 | Offensive | Insulte ou contenu inapproprié |

**Métriques :**
| Métrique | Cible | Évaluation |
|----------|-------|-----------|
| Absence de crash | 0 exception sur 4 cas | Automatisée |
| État session intact | Inchangé après chaque entrée | Automatisée |
| Qualité de la redirection | ≥ 4/5 | Humaine (double évaluation) |

**Conditions d'objectivité :**
| Mesure | Détail |
|--------|--------|
| Fixture d'état initial | Reset complet BD + mémoires avant l'exécution |
| Corpus standardisé | Les 4 entrées sont fixées et versionnées — mêmes entrées à chaque run |
| Double évaluation humaine | Deux évaluateurs indépendants — accord ≥ 80 % requis |

**Tests correspondants :** `tests/eval/test_s8_robustness.py`
| Test | Vérifie |
|------|---------|
| `test_s8_no_crash_on_unexpected_input[hors-domaine]` | 0 exception sur entrée hors-domaine |
| `test_s8_no_crash_on_unexpected_input[ambiguë]` | 0 exception sur entrée ambiguë |
| `test_s8_no_crash_on_unexpected_input[vide]` | 0 exception sur entrée vide |
| `test_s8_no_crash_on_unexpected_input[offensive]` | 0 exception sur entrée offensive |
| `test_s8_session_state_intact_after_input[hors-domaine]` | État de session inchangé après entrée hors-domaine |
| `test_s8_session_state_intact_after_input[ambiguë]` | État de session inchangé après entrée ambiguë |
| `test_s8_session_state_intact_after_input[vide]` | État de session inchangé après entrée vide |
| `test_s8_session_state_intact_after_input[offensive]` | État de session inchangé après entrée offensive |
| `test_s8_empty_input_does_not_trigger_routing` | Message vide ignoré sans déclencher le routage |
| `test_s8_all_4_cases_covered` | Le corpus couvre bien les 4 types standardisés |
| `test_s8_human_evaluation_framework` | Documente les réponses pour évaluation de la qualité de redirection |

---

## Cadre d'interprétation des résultats

### Baseline de comparaison

Les résultats du système doivent être lus en regard d'une baseline, sans quoi les chiffres n'ont pas de sens :

| Métrique | Baseline (système naïf) | Cible Open TutorAI | Interprétation |
|----------|------------------------|-------------------|---------------|
| Recall@5 | ~0,20 (récupération aléatoire dans 10 docs) | ≥ 0,80 | L'écart avec 0,20 mesure la valeur ajoutée de l'indexation sémantique |
| Faithfulness | ~0,50 (LLM sans RAG) | ≥ 0,90 | L'écart avec 0,50 mesure la réduction des hallucinations apportée par le RAG |
| MRA | 0 % (pas de mémoire) | 100 % | Tout écart en dessous de 100 % signale une perte d'information pédagogique |
| CR | 1,0 (pas de compression) | ≤ 0,30 | Le gain par rapport à 1,0 mesure l'efficacité du pipeline de résumé |
| IRR | ~40 % (résumé naïf, premiers tokens) | ≥ 0,80 | L'écart avec 40 % mesure la valeur ajoutée du résumé intelligent |
| LRA | ~17 % (routeur aléatoire, 6 agents) | ≥ 80 % | L'écart avec 17 % mesure la valeur ajoutée du LLM sur le routage |
| TCR | 0 % (pas d'orchestration) | ≥ 95 % | Mesure la fiabilité end-to-end de l'architecture agentique |
| Learning Gain | 0 (pas d'adaptation) | ≥ 0,30 | Seuil de gain « moyen » (Hake, 1998) |

### Interprétation graduée (au-delà du PASS / FAIL)

Un résultat ne doit pas être lu comme binaire. Chaque métrique admet une lecture graduée :

**Recall@5**
| Valeur | Interprétation |
|--------|---------------|
| ≥ 0,80 | Les documents pertinents sont correctement récupérés — comportement attendu |
| 0,60–0,79 | Récupération partielle — vérifier la qualité de l'indexation et des embeddings |
| < 0,60 | Défaillance de la récupération — le pipeline RAG ne trouve pas les bonnes sources |

**Faithfulness**
| Valeur | Interprétation |
|--------|---------------|
| ≥ 0,90 | Réponse ancrée dans les sources — comportement attendu |
| 0,70–0,89 | Hallucinations ponctuelles — surveiller les affirmations non sourcées |
| < 0,70 | Hallucinations fréquentes — le LLM génère hors des sources récupérées |

**Memory Retrieval Accuracy**
| Valeur | Interprétation |
|--------|---------------|
| 100 % | Mémoire parfaitement fiable — comportement attendu |
| 80–99 % | Pertes ponctuelles — vérifier quels champs sont manquants et pourquoi |
| < 80 % | Défaillance structurelle — le système ne peut pas assurer un suivi cohérent |

**Compression Ratio**
| Valeur | Interprétation |
|--------|---------------|
| ≤ 0,20 | Compression très agressive — surveiller l'IRR, risque de perte d'information |
| 0,21–0,30 | Zone cible — équilibre entre réduction et rétention |
| > 0,30 | Compression insuffisante — risque de dépassement de la fenêtre de contexte |

**LLM Routing Accuracy**
| Valeur | Interprétation |
|--------|---------------|
| ≥ 80 % | Le LLM pilote efficacement l'orchestration |
| 60–79 % | Le LLM est instable — le fallback compense, mais la qualité de décision est dégradée |
| < 60 % | Le LLM n'est pas fiable pour le routage — basculer sur le routage déterministe |

**Learning Gain**
| Valeur | Interprétation |
|--------|---------------|
| ≥ 0,70 | Gain élevé — dispositif très efficace (Hake, 1998) |
| 0,30–0,69 | Gain moyen — objectif atteint |
| < 0,30 | Gain faible — l'adaptation pédagogique n'est pas suffisamment efficace |

### Trade-offs entre métriques

Certaines métriques sont en tension : améliorer l'une peut dégrader l'autre. Ces trade-offs doivent être analysés conjointement :

| Trade-off | Description |
|-----------|-------------|
| **Recall@k ↔ Faithfulness** | Augmenter k améliore le Recall mais injecte des documents moins pertinents dans le contexte du LLM, ce qui peut diluer la Faithfulness. Calibrer k selon le seuil de Faithfulness observé. |
| **CR ↔ IRR** | Une compression plus agressive (CR plus faible) augmente le risque de perte d'information (IRR plus faible). Si CR ≤ 0,20 et IRR < 0,80, le pipeline sacrifie la qualité pour la taille. |
| **LRA ↔ Latence** | Un LLM plus puissant améliore la LRA mais augmente la latence. Si la latence dépasse 3 s, évaluer si le gain en LRA justifie la dégradation de l'expérience utilisateur. |
| **MRA ↔ Scalabilité** | Une mémoire exhaustive (MRA = 100 %) implique de stocker et récupérer davantage d'informations — surveiller l'impact sur les temps de réponse à mesure que le profil apprenant s'enrichit. |
