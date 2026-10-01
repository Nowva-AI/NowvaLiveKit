# Plan — Soulevé de terre conventionnel (deadlift) — v1

Statut : proposition, à valider par Ambaka. Version 1.

## 0. Décisions déjà prises (ne pas rediscuter)

| Décision | Conséquence pour le plan |
|---|---|
| On sort du « squat only » de `CLAUDE.md` pour le deadlift | Le deadlift est un chantier autorisé, mais il ne doit rien coûter au squat |
| **Le squat ne doit jamais être modifié et doit rester séparé du deadlift** | Aucune ligne des fichiers squat n'est modifiée ; le deadlift vit dans son propre package ; les rares fichiers partagés ne reçoivent que des ajouts derrière un drapeau, et une batterie de tests « golden master » prouve que le squat produit exactement les mêmes sorties |
| Deadlift **conventionnel uniquement**, pas de sumo | « sumo » est refusé poliment par l'agent vocal ; aucune logique sumo |
| Dos rond : OK pour des indicateurs indirects (proxys) | On ne prétend jamais mesurer la colonne ; on mesure la « perte de position » |
| Les caméras du rack voient la barre au sol | Il faut construire la détection et le suivi 3D de la barre au sol |
| Apprentissage du LLM : hors périmètre | On continue juste à enregistrer les données (avec l'exercice) pour plus tard |
| **Piège de l'axe Y** | Voir §2 — règle bloquante pour tout le code deadlift |

## 1. Objectif produit et critère de succès

La boucle démo YC, version deadlift :

> L'athlète dit « on fait du deadlift » → Nova le guide pour se placer → il tire → le système détecte une faute mesurable (ex. barre trop loin du milieu du pied, hanches qui montent avant les épaules) → Nova donne **un** cue chiffré au bon moment (au sol, avant la rep suivante) → la rep suivante est corrigée → récap de série.

Critère de succès v1 (mesuré sur des données réelles annotées, §8) :

| Métrique | Cible |
|---|---|
| Comptage des reps | ≥ 99 % de reps correctement comptées, 0 rep fantôme sur les séries propres |
| Précision par faute v1 | ≥ 0,85 (une fausse correction détruit la confiance plus qu'un oubli) |
| Rappel par faute v1 | ≥ 0,70 |
| Fausses corrections sur reps propres | ≤ 1 pour 10 reps |
| Position 3D de la barre | erreur ≤ 1,5 cm (avant/arrière), ≤ 1 cm (hauteur) |
| Barre ↔ milieu du pied au départ | erreur ≤ 2 cm |
| Latence cue | identique au squat (cue pré-enregistré < 50 ms après la décision) |
| Squat | golden master identique, bit pour bit |
| Edge | budget total ≤ 33 ms/frame sur Jetson Orin Nano Super (à mesurer, §10) |

## 2. Conventions et piège de l'axe Y (règle bloquante)

Le code utilise partout un repère monde **Y vers le bas**, en mètres : un Y **plus grand = plus bas**.
X = gauche du sujet, **+Z = dos du sujet** (donc « devant » = −Z). Sources : `src/biomechanics/triangulation/person_calibration.py:5`, `triangulator.py:2`, `utils/foot_contact.py:32`, `kinematics/analytical_ik.py:449`.

Deux documents se contredisent avec le code et **ne doivent pas être suivis** :
- `.claude/rules/biomechanics.md` : « Larger Y = higher » → **faux**.
- `README.md:325` : « Z-forward (toward camera) » → **faux** (avant = −Z).

Règles pour tout le code deadlift :
1. Un seul module `src/biomechanics/deadlift/frame.py` définit les conversions, et rien d'autre ne fait de soustraction de Y ou de Z à la main :
   - `height_above_floor_m(point_y, floor_y) = floor_y - point_y` (positif au-dessus du sol)
   - `forward_offset_m(point_z, reference_z) = reference_z - point_z` (positif si le point est devant la référence)
2. Les noms disent le sens : `bar_height_above_floor_m`, `bar_forward_of_midfoot_cm`, jamais `bar_y`.
3. Tests de signe obligatoires (squelette synthétique debout et en position de départ) : la hanche debout est à > 0,7 m au-dessus du sol ; la barre monte pendant le tirage ; une barre posée devant le pied donne un `forward` positif.
4. On travaille dans le repère **monde** (`PreIKResult.analysis_world`), pas dans le squelette recentré sur les hanches, pour tout ce qui touche au sol et à la barre.

## 3. Isolation du squat (contrainte dure)

### 3.1 Architecture : un pipeline deadlift séparé

Constat (`src/main.py:452`) : `main.py` lance un sous-processus `pipeline_process.py` avec le nom de l'exercice. C'est le point de bifurcation le plus propre.

```
main.py:452  ──  exercice deadlift ET drapeau activé ?
                   ├── non → pipeline_process.py (squat, inchangé)
                   └── oui → biomechanics/deadlift/process.py (nouveau)
```

- `pipeline_process.py` et `pipeline.py` : **0 ligne modifiée**.
- Le package `src/biomechanics/deadlift/` contient tout : boucle de session, sous-classe de pipeline, détection de barre 3D, machine à états des reps, métriques, règles, diagnostic, session tracker, cues, textes.
- Réutilisation **par import uniquement** (jamais de modification) : `CameraCalibrationSession`, `DLTTriangulator`, `build_preik_chain`, `AnalyticalIK`, `SegmentLengthEstimator`, `IPCBridge`, `BarbellDetector`, types de `utils/types.py`, `DiagnosisResult`.
- `DeadliftPipeline` hérite de `BiomechanicsPipeline` mais **redéfinit entièrement `process_frame`** et reçoit une copie de config avec `bilstm.enabled = False` (le BiLSTM est entraîné sur des squats et rejette les reps deadlift : `pipeline.py:243-259, 815-854`).
- Les helpers privés de `pipeline_process.py` dont on a besoin (HUD, throttle, poll IPC) sont importés s'ils sont au niveau module, sinon recopiés dans le package deadlift avec un commentaire d'origine. On accepte cette duplication : c'est le prix du « jamais toucher au squat ». Une factorisation éventuelle plus tard se fera uniquement avec l'accord d'Ambaka et sous golden master.
- Le profil `profiles/deadlift.py` actuel (placeholder jamais testé, sa `BackRoundingRule` signalerait l'inclinaison normale du deadlift) n'est plus utilisé. Il est laissé en place pour ne pas toucher au registre partagé ; le package deadlift a son propre profil.

### 3.2 Drapeau de fonctionnalité

`NOWVA_ENABLE_DEADLIFT` (défaut : `false`). Drapeau éteint ⇒ chaque fichier partagé se comporte exactement comme aujourd'hui (y compris le texte des prompts). Il reste éteint en production tant que les critères §1 ne sont pas atteints.

### 3.3 Liste exhaustive des fichiers partagés touchés (ajouts seulement, derrière le drapeau)

| Fichier | Changement | Effet sur le squat |
|---|---|---|
| `src/main.py` (~l. 452) | Choix du script de sous-processus | Aucun : chemin squat identique |
| `src/agent/agents/prompts/main_menu_prompt.py:26` | Ligne « only squats » remplacée par « squats et deadlift conventionnel » **seulement si drapeau** | Aucun drapeau éteint |
| `src/agent/agents/shared/helpers.py:94-99` | Ajout d'alias (« Barbell Conventional Deadlift », « conventional deadlift ») | Aucun : alias squat inchangés |
| `src/biomechanics/calibration.py:22-34` | Ajout d'entrées `movement_pattern = "conventional_deadlift"` (clé distincte du `hip_hinge` du RDL) | Aucun |
| `src/agent/agents/quickExerciseAgent.py:211` | Passe `exercise=` au `TeachingAgent` | Squat : valeur passée = « squat », identique au défaut |
| `src/agent/agents/teaching_agent.py`, `calibration_agent.py`, `workout_agent.py` | Retour anticipé vers un module de textes deadlift si exercice = deadlift | Code squat non réindenté, non modifié |
| `src/agent/services/coaching_service.py:129, 159-160, 825-935` | `BiomechanicsRecorder(exercise=...)`, requêtes de progression filtrées par exercice, branche de récap deadlift | Squat : `exercise="squat"` = défaut actuel |
| `src/agent/services/coaching_orchestrator.py:1241-1251, 1407, 1510` | Branche de texte de récap deadlift | Aucun |
| `src/agent/services/coaching_constants.py` | Ajout de clés de cue deadlift (`CUE_TEXT_MAP`, `CUE_DISPLAY_LABELS`, `PREEMPTIVE_TEXT`) | Aucun (ajouts) |
| `scripts/tools/generate_cue_audio.py` | Ajout des cues deadlift | Aucun (ajouts) |

Tout autre fichier partagé qui devrait changer est d'abord signalé à Ambaka.

### 3.4 Filet de sécurité pour le squat (Jalon 0, avant toute ligne de deadlift)

1. **Golden master squat** — `tests/test_biomechanics/test_squat_golden.py` :
   - Rejoue des scénarios fixes via le harnais multi-caméra existant de `tests/test_biomechanics/test_pipeline.py:329-392` (`_FakeProvider`, `_FakeClock`, `world_squat_points`, `squat_depth_profile`) : série propre, valgus, inclinaison, rep trop haute (`go_deeper`), perte de détection, asymétrie.
   - Chemin mono-caméra via `estimate_both` simulé ; chemin BiLSTM via un faux `_bilstm` déterministe.
   - Pour chaque frame : `model_dump(mode="json")` sans `latency_ms`, NaN → null. Puis passage dans `SessionTracker` + `IPCBridge` (client IPC simulé) et `force_end_set` : on enregistre tous les messages IPC, y compris `diagnosis_complete`.
   - Comparaison avec des JSON dans `tests/test_biomechanics/fixtures/golden/`, régénérés uniquement avec une variable d'environnement explicite. Flottants arrondis à 1e-9.
   - Lancé deux fois : drapeau éteint **et** allumé — les sorties squat doivent être identiques dans les deux cas.
2. **Golden master des prompts squat** : snapshot du texte des agents squat (menu avec drapeau éteint, teaching, calibration, workout).
3. **Test des alias** : chaque alias squat se résout toujours vers `SquatProfile` ; « sumo deadlift » ne lance jamais le pipeline deadlift.
4. **Manifeste de gel** : `tests/test_squat_freeze.py` vérifie l'empreinte SHA-256 des fichiers 100 % squat (`profiles/squat.py`, `diagnosis/**`, `pipeline.py`, `pipeline_process.py`, `faults/rules/{depth,knee_valgus,forward_lean,heel_rise,symmetry,bar_tilt_asymmetry}.py`, `faults/hip_position_counter.py`, `coaching/{session_tracker,ipc_bridge,cue_cache}.py`). Toute modification fait échouer le test ; mettre à jour le manifeste exige l'accord explicite d'Ambaka.
5. **Rejeu de vraies séances squat** enregistrées dans `user_test_runs/` (machine d'Ambaka) avant/après chaque jalon : sorties identiques.
6. **CI** : il n'existe aucune CI aujourd'hui (pas de `.github/workflows`). On propose un workflow GitHub Actions qui installe `requirements.lock` et lance `PYTHONPATH=src pytest tests/test_biomechanics -x` sur chaque PR. À valider avec Ambaka (infra nouvelle).

Preuve que le filet fonctionne : modifier volontairement un seuil squat en local doit faire échouer le golden master.

## 4. Le mouvement : phases et machine à états

### 4.1 Phases d'une rep de deadlift conventionnel

`APPROCHE` (debout) → `PLACEMENT` (penché à la barre, immobile) → `DÉCOLLAGE` (la barre quitte le sol) → `PASSAGE DES GENOUX` → `LOCKOUT` (debout, hanches et genoux tendus) → `DESCENTE` → `AU SOL` (barre posée) → rep suivante (arrêt complet ou touch-and-go).

Différence clé avec le squat : la rep **commence en bas**, avec un temps d'arrêt statique au sol. Ce temps de placement est le meilleur moment pour corriger (le lifter est immobile et peut ajuster), alors que pendant le tirage (≈ 1 s, effort maximal) il ne peut pas réagir à un cue.

### 4.2 Signal de rep

Priorité :
1. **Hauteur du centre de la barre au-dessus du sol** (barre 3D, §6).
2. Repli : **hauteur du milieu des poignets au-dessus du sol** (les mains tiennent la barre).
3. Dernier recours : hauteur de la hanche au-dessus du sol (`FootState.hip_height_above_floor_cm` n'est pas réutilisable directement — le deadlift calcule la sienne, §6.5).

On n'inverse **pas** le signal de hanche comme le placeholder actuel : la hauteur de barre monte quand on tire, ce qui est lisible et évite les erreurs de signe.

### 4.3 Machine à états (`deadlift/rep_counter.py`, nouvelle)

| Transition | Condition (valeurs initiales, à valider §8) |
|---|---|
| `APPROCHE → PLACEMENT` | mains à moins de 10 cm au-dessus de la hauteur de repos de la barre, vitesse < 0,05 m/s pendant ≥ 0,3 s |
| `PLACEMENT → TIRAGE` | barre > hauteur de repos + 3 cm et vitesse > 0,10 m/s |
| `TIRAGE → LOCKOUT` | vitesse < 0,05 m/s, barre ≥ hauteur de repos + 30 cm (ou ≥ 60 % de la hauteur de lockout attendue d'après l'anthropométrie) |
| `LOCKOUT → DESCENTE` | vitesse < −0,10 m/s |
| `DESCENTE → AU SOL` | barre ≤ hauteur de repos + 3 cm → **rep analysée** |
| `DESCENTE → TIRAGE` | touch-and-go : la vitesse repasse > 0 à moins de 5 cm du sol |
| `TIRAGE → PLACEMENT` sans lockout | rep ratée : événement « rep échouée », pas de comptage |

- Le compte vocal (« One! ») part à l'entrée en `LOCKOUT` ; l'analyse complète de la rep part à `AU SOL`.
- La hauteur de repos de la barre est mesurée à chaque série (médiane pendant `PLACEMENT`), donc elle s'adapte à la taille des disques.
- Barre lâchée depuis le lockout (descente très rapide) : rep comptée + faute « descente non contrôlée ».
- Petits ajustements de la barre au sol (< 3 cm) : ignorés.
- Fin de série : barre au sol et lifter debout ou hors champ pendant `set_timeout_seconds` (30 s, config existante), ou « j'ai fini » à la voix.
- Porte de départ : le lifter arrive debout, donc la porte debout existante peut servir **une fois par série** ; elle n'est jamais réarmée entre deux reps.

## 5. Fautes, mesures et seuils

Toutes les mesures sont dans le plan sagittal du lifter (calculé à partir de la ligne des hanches dans le repère monde), sauf mention contraire. Les seuils sont des **valeurs initiales d'ingénieur**, à fixer sur données réelles (§8) avant toute démo. Trois niveaux toujours : léger / modéré / sévère.

### 5.1 Fautes v1 (démo)

| # | Faute | Mesure | Seuils initiaux | Moment du cue | Cue |
|---|---|---|---|---|---|
| F1 | **Barre trop loin du milieu du pied** au départ | `bar_forward_of_midfoot_cm` sur la dernière 0,3 s de `PLACEMENT` ; milieu du pied = milieu talon ↔ pointe (`LEFT/RIGHT_HEEL`, `LEFT/RIGHT_FOOT_INDEX`) | 3 / 5 / 8 cm | Pendant le placement (si immobile ≥ 0,5 s) ou au sol avant la rep suivante | « Avance les pieds, barre au-dessus du milieu du pied » + nombre de cm |
| F2 | **Hanches qui montent avant les épaules** (« stripper pull ») | Variation de l'angle du tronc (sagittal, signé) entre le décollage et le passage de la barre à hauteur des genoux ; contrôle croisé : déplacement vertical hanche / épaule sur la même phase | 8° / 12° / 18° ; rapport > 1,3 | Au sol, avant la rep suivante | « Poitrine et hanches montent ensemble — pousse le sol » |
| F3 | **Barre qui s'éloigne du corps** | Écart avant/arrière max du centre de barre par rapport à sa position de départ pendant le tirage ; distance barre ↔ genou au passage des genoux | 3 / 5 / 8 cm | Au sol | « Garde la barre collée aux jambes » |
| F4 | **Lockout incomplet** | Au point haut : extension de hanche (angle tronc ↔ cuisse, sagittal) et flexion de genou | déficit > 10° / 15° / 25° | Au sol | « Serre les fessiers, finis debout » |
| F5 | **Hyperextension au lockout** (se pencher en arrière) | Tronc au-delà de la verticale vers l'arrière au point haut (angle signé) | 8° / 12° / 18° | Au sol | « Grandis-toi, ne te penche pas en arrière » |
| F6 | **Asymétrie** | Déplacement latéral du bassin par rapport au milieu des pieds pendant le tirage ; inclinaison de la barre (différence de hauteur entre ses deux extrémités, barre 3D) ; rotation du tronc (champs IK existants) | 2 / 4 / 6 cm ; 2 / 4 / 6 cm ; 5° / 8° / 12° | Au sol | « Pousse pareil dans les deux pieds » |
| F7 | **Bras pliés** pendant le tirage | Flexion du coude (`elbow_flexion_l/r`, IK existante) entre décollage et lockout | 15° / 25° / 35° | Au sol | « Bras tendus, les bras sont des cordes » |

### 5.2 Fautes v1.1 (après la démo, si les données le permettent)

| Faute | Mesure | Remarque |
|---|---|---|
| Départ avec hanches trop basses (« squatter le deadlift ») ou trop hautes | Hauteur de hanche au placement vs modèle géométrique (§5.4) | Fait aussi partie du diagnostic de F2 |
| Arrachage de la barre (pas de « mise en tension ») | Pic d'accélération de la barre au décollage ; petite descente des hanches juste avant | Demande une barre 3D précise |
| Descente non contrôlée / rebond en touch-and-go | Vitesse de descente de la barre, impact | |
| Genoux qui avancent avant que la barre passe les genoux à la descente | Ordre hanches/genoux à la descente | |
| Écartement des pieds / prise trop large | Distance chevilles vs largeur de hanches ; poignets vs tibias | Fautes de placement, faciles à mesurer |
| Genoux qui rentrent pendant le tirage | Valgus (métrique IK existante, réutilisée par import) | Moins fréquent au deadlift qu'au squat |
| Talons qui décollent / poids sur les orteils | Nouvelle instance de la logique de `foot_contact` dans le package deadlift | |
| Perte de vitesse de barre dans la série (fatigue) | Vitesse concentrique moyenne par rep (barre 3D) | Base du diagnostic « charge trop lourde » ; aussi utile pour le VBT |

### 5.3 Dos rond : proxys honnêtes

Les keypoints (COCO 21 avec talons) n'ont aucun point sur la colonne : la ligne épaule ↔ hanche reste droite que le dos soit rond ou non. Donc :
- **On ne mesure pas le dos rond et on ne le dit jamais à l'utilisateur.** On mesure une « perte de position » composée de F2 (hanches qui montent), F3 (barre qui s'éloigne) et, en expérimental, du raccourcissement de la distance épaule ↔ hanche par rapport à la longueur mesurée debout (la flexion de la colonne raccourcit cette corde).
- Le signal expérimental n'est activé que si les données réelles (§8) montrent qu'il sépare les reps « dos rond » (annotées par un coach) des reps propres, au-delà du bruit (bruit Kalman ≈ 3 mm).
- Le cue reste comportemental (« dos plat, gaine-toi, poitrine et hanches ensemble »), jamais médical. Aucune affirmation du type « ton dos est en sécurité ».
- La règle `BackRoundingRule` actuelle n'est pas utilisée : elle mesure la variation d'inclinaison du tronc, ce qui est le mouvement normal du deadlift.

### 5.4 Modèle géométrique du placement attendu (cœur de l'IP)

Équivalent deadlift de `expected_trunk_lean_geometric` du squat : prédire, pour **ce corps-là**, la position de départ idéale, puis comparer.

Contraintes du modèle de coaching classique : barre au-dessus du milieu du pied, épaules juste devant la barre, bras verticaux, tibias en contact avec la barre.

- Entrées : longueurs tibia, fémur, torse, bras (épaule → poignet), pied ; hauteur de repos de la barre (mesurée) ; position du milieu du pied.
- `SegmentLengthEstimator` ne mesure **pas** les bras (`utils/segment_lengths.py:53-64`). Le package deadlift ajoute son propre estimateur de longueur de bras (épaule → coude → poignet), sans toucher au fichier partagé.
- Résolution : épaule fixée au-dessus de la barre à hauteur `barre + bras` ; cheville fixée ; chaîne cheville → genou → hanche → épaule avec longueurs connues ⇒ un seul degré de liberté (angle du tibia), choisi pour que le tibia touche la barre. Recherche 1D, quelques microsecondes.
- Sorties : hauteur de hanche attendue, angle du tronc attendu, angle du genou attendu.
- Utilisation : diagnostic « hanches trop basses / trop hautes de X cm », seuils de F2 relatifs à ce placement, et plus tard le « fantôme » de la position idéale.
- Validation : tests unitaires sur géométrie connue ; contrôle de cohérence (fémurs longs / bras courts ⇒ dos plus horizontal) ; comparaison avec les placements réels annotés « bons » par le coach.

### 5.5 Personnalisation des seuils

Différent du squat (`SquatProfile.apply_baseline` fixe les seuils à « pic observé + 5/10/15° », ce qui peut normaliser une faute) :
- Une série d'échauffement (barre vide / charge légère) mesure le **bruit** de chaque métrique chez cet utilisateur.
- Seuil = max(seuil population, k × bruit mesuré), plafonné par une limite de sécurité qu'aucune calibration ne peut dépasser.
- Stockage par utilisateur sous `movement_pattern = "conventional_deadlift"`. Le chemin deadlift n'utilise jamais le repli `or "squat"` de `pipeline_process.py:1073`, donc il ne peut pas écraser la calibration squat.

## 6. Barre au sol : détection et suivi 3D

### 6.1 Existant

- `BarbellDetector` (`barbell_tracking/detector.py`) : YOLO11n-pose, une boîte + 2 keypoints (extrémités), ne renvoie que la meilleure détection.
- Les poids `models/barbell_keypoints.pt` ne sont pas dans le repo ; aucune trace de leur entraînement ni des données. Rien n'indique qu'il a vu des barres au sol chargées.
- `barbell_tracking.enabled` vaut `False` par défaut (`config.py:146`) ; en multi-caméra, la détection live ne tourne que sur la caméra principale, en 2D (`pipeline.py:613-624`).
- Les extrémités de barre ne sont triangulées en 3D que pendant la calibration caméra, pour l'échelle, puis jetées (`person_calibration.py:599-601, 852, 968`).
- `BarPathRule` mélange gauche-droite (avec détection) et avant-arrière (repli poignets) : non réutilisée.

### 6.2 Définition des keypoints

Pour une barre chargée, les extrémités du manchon sont souvent cachées. Proposition : 2 keypoints = **centre du disque extérieur côté gauche et côté droit** (sur l'axe de la barre, face extérieure). Un disque de 45 cm est grand, contrasté et visible de face comme à 45°. À confirmer sur les premières images réelles (J1) avant d'annoter en masse.

### 6.3 Données et entraînement

1. Récupérer les poids actuels sur la machine d'Ambaka et les évaluer tels quels sur des images de barre au sol (J1). Décision : affiner ce modèle ou repartir de YOLO11n-pose de base.
2. Capture (§8.3, en même temps que les séances deadlift) : barre au sol, en tirage, au lockout, barre rangée dans les J-hooks (distracteur), barre vide / disques bumper / fonte, plusieurs couleurs, éclairages, lifters, tenues. Les 3 caméras.
3. Annotation : 2 000–3 000 images (≈ équilibrées entre caméras et phases) avec un outil local (CVAT ou Label Studio auto-hébergé). Négatifs inclus.
4. Entraînement YOLO11n-pose (Ultralytics, déjà en dépendance), export ONNX puis TensorRT FP16 pour le Jetson.
5. Évaluation par lifter (aucun lifter à la fois en train et en test) : rappel ≥ 95 % sur barre au sol, erreur keypoint en pixels, faux positifs sur barre rangée.

### 6.4 Exécution en série (nouveau module `deadlift/bar_tracker_3d.py`)

1. Détection sur **les 3 vues**, en lot (comme le batch RTMPose), avec **toutes** les détections (pas seulement la meilleure) — wrapper deadlift autour de `BarbellDetector`, sans modifier la classe.
2. Association entre vues : on garde la barre dont la triangulation a la plus petite erreur de reprojection et qui est proche des mains / pieds du lifter (élimine la barre rangée).
3. Étiquetage gauche/droite par la ligne des épaules (même principe que `_label_bar_ends_by_shoulders`, recopié).
4. Triangulation DLT des 2 extrémités (petite implémentation dans le package, ≈ 40 lignes ; les fonctions de `triangulator.py` sont privées) ; rejet si erreur de reprojection > seuil.
5. Kalman 3D à vitesse constante par extrémité ; prédiction pendant les trous courts.
6. Sortie `BarState3D` : centre (monde), hauteur au-dessus du sol, avant/arrière par rapport au milieu du pied, inclinaison (cm et °), vitesse (m/s), source (`bar` / `wrist_proxy`), confiance.

### 6.5 Référence du sol

`foot_contact.py` ne publie pas le sol (ancres privées) et est un fichier partagé. Le package deadlift calcule son propre sol :
- pendant `APPROCHE` (debout immobile) : Y des talons et pointes dans le repère monde, plus un décalage keypoint → sol (≈ 2–3 cm, à mesurer) ;
- contrôle croisé : barre au repos = rayon du disque au-dessus du sol ;
- réinitialisé quand la calibration caméra change (même logique que `pipeline.on_calibration_changed`).

### 6.6 Repli sans barre

Si la barre n'est pas suivie : milieu des poignets, avec un décalage poignet → barre appris par utilisateur quand les deux sont visibles. Les fautes qui exigent la barre (F1, F3, inclinaison de F6) sont marquées `source = wrist_proxy` et leurs seuils sont élargis ; si la confiance est trop basse, la faute n'est pas émise (pas de cue sur une mesure douteuse).

### 6.7 Coût edge

YOLO11n-pose à 640 px × 3 vues est le principal coût ajouté. À mesurer sur Jetson (§10). Leviers dans l'ordre : TensorRT FP16, entrée 480 px, détection une frame sur deux avec prédiction Kalman entre, recadrage autour des mains/pieds, seulement les 2 caméras à 45°.

## 7. Diagnostic après la série (graphe statique, comme le squat à ses débuts)

### 7.1 Principe

Même méthode que le squat : symptômes → causes candidates → probabilité de départ écrite à la main × test de preuve → normalisation avec fuite (`UNEXPLAINED_LEAK_SCORE`) → noisy-OR. Sortie dans le **même schéma `DiagnosisResult`** pour que le côté voix le lise sans changement de format.

Mais dans un module séparé `src/biomechanics/deadlift/diagnosis/` : `symptoms.yaml`, `causes.yaml`, `evidence_tests.py`, `parameter_deltas.py`, `engine.py`. Le moteur squat charge ses YAML au moment de l'import via des globales (`diagnosis/graph/loader.py:114-116`), donc on ne peut pas le réutiliser sans le modifier : le moteur deadlift réimplémente les ≈ 150 lignes de calcul (duplication assumée, §3.1).

Sorties gardées simples (garde-fou n°2) : par hypothèse, `cause_id`, niveau, score, **un** delta chiffré, une phrase.

### 7.2 Symptômes v1

`hips_rise_early` (F2), `bar_forward_at_setup` (F1), `bar_drift_in_pull` (F3), `incomplete_lockout` (F4), `lockout_overextension` (F5), `lateral_shift` (F6), `bent_arms` (F7). Valeur attendue personnalisée par le modèle géométrique (§5.4) quand c'est pertinent (ex. angle de tronc attendu).

### 7.3 Causes v1 (priors à la main, documentés)

| Cause | Niveau | Delta chiffré | Impliquée par |
|---|---|---|---|
| `setup_bar_too_far` | 1 | « avance de X cm » | F1, F3, F2 |
| `setup_hips_too_low` | 1 | « hanches X cm plus haut » | F2 |
| `setup_hips_too_high` | 1 | « hanches X cm plus bas » | F3 |
| `lats_not_engaged` | 1 | — | F3 |
| `slack_not_pulled` | 1 | — | F2 (v1.1 : accélération) |
| `glutes_not_finishing` | 1 | — | F4 |
| `overextension_habit` | 1 | — | F5 |
| `uneven_stance_or_grip` | 1 | « décale la prise de X cm » (poignets vs centre de barre) | F6 |
| `weight_too_heavy` | 2 | « baisse d'environ 10 % » | dégradation au fil de la série, perte de vitesse |
| `weak_off_floor` | 3 | — | F2 malgré un bon placement |
| `limited_hip_hamstring_mobility` | 3 | — | placement impossible à atteindre |
| `unilateral_weakness` | 3 | — | F6 persistant |
| `anthropometric_context` | 0 | — | « fémurs longs / bras courts : dos plus horizontal attendu » |

### 7.4 Score de rep deadlift

5 dimensions dans [0, 1], poids statiques : placement (25 %), coordination hanches/épaules (25 %), trajectoire de barre (20 %), lockout (15 %), symétrie (15 %). Pas de dimension « profondeur ».

### 7.5 Fantôme (correction visuelle)

Le correcteur de keypoints du squat (`diagnosis/keypoint_corrector.py`) est spécifique au squat : non réutilisé. Optionnel pour la démo : afficher le placement idéal calculé en §5.4 comme fantôme statique. Ne bloque pas la v1.

## 8. Données

### 8.1 Connaissance statique (point de départ, comme le squat)

`docs/deadlift/KNOWLEDGE.md` : phases, définition exacte de chaque métrique (avec repère et signe), seuils initiaux + justification, textes des cues, modèle géométrique. Sources : modèle de placement du coaching classique (barre au-dessus du milieu du pied, épaules au-dessus de la barre, bras verticaux), manuels de référence en préparation physique, revue par Ambaka et idéalement par un coach de force externe. Chaque seuil porte la mention « à valider sur données ».

### 8.2 Données synthétiques (simulateur)

Le simulateur à vérité terrain existe pour le squat : `.claude/preik-audit/harness/preik_harness/` (`body.py`, `cameras.py`, `detector.py`, `runner.py`, `metrics.py`).
- Corriger les chemins en dur (`REPO_SRC="/Users/naiahoard/..."` dans `__init__.py:13`) par un chemin relatif.
- Ajouter un générateur deadlift (≈ 200–300 lignes) : départ au sol, mains liées à la barre, `pose_from_params` réutilisé, barre au sol (aujourd'hui `barbell.py` la met sur le dos), occultation par les disques.
- Scénarios avec vérité terrain : propre, barre loin, hanches qui montent, barre qui s'éloigne, lockout incomplet, hyperextension, décalage de hanches, bras pliés, touch-and-go, barre lâchée, rep ratée ; plusieurs morphologies.
- Le runner prend le profil en paramètre (aujourd'hui « Barbell Back Squat » en dur, `runner.py:404`).
- Usage : tests unitaires et d'intégration, sanity des seuils, robustesse au bruit. **Les données synthétiques ne servent pas à fixer les seuils finaux.**

### 8.3 Données réelles (indispensables)

Protocole de capture :
- **Round 1 (équipe, semaine 1–2)** : 2–3 personnes, barre vide et charges légères. Reps propres + fautes volontaires sans danger (barre loin, hanches qui montent, lockout mou, se pencher en arrière, décalage de hanches, bras pliés). Le dos rond n'est montré qu'avec barre vide ou bâton (sécurité).
- **Round 2 (pilote)** : 10–15 lifters de morphologies variées (1,55–1,95 m, différents ratios fémur/torse/bras), débutants à confirmés, charges modérées (RPE ≤ 8), séries naturelles.
- Ce qu'on enregistre : vidéo brute synchronisée des 3 caméras, fichier de calibration, sorties du pipeline (`user_test_runs/`), métadonnées (charge, type de disques, taille, envergure, longueurs de segments mesurées au mètre ruban sur un sous-ensemble), consentement écrit.
- Vérité terrain de position : sur un sous-ensemble, scotch au sol et gabarit de pied pour placer la barre à 0 / 3 / 6 / 10 cm devant le milieu du pied (valide F1 et la précision 3D de la barre).
- Annotation : chaque rep annotée par un coach (Ambaka ou externe) — faute présente / niveau — avec l'outil de rejeu HTML existant adapté au deadlift ; 20 % annotés par 2 personnes pour mesurer l'accord.
- Volume cible v1 : ≥ 300 reps annotées, ≥ 50 exemples positifs par faute v1 ; 2 000–3 000 images annotées pour la barre.
- Découpage par lifter (validation sur des lifters jamais vus).
- Données stockées en local, anonymisées, avec consentement.

### 8.4 Ce que les données doivent vérifier avant la démo

1. Qualité de la pose en position penchée (visage vers le sol, bras devant les genoux vus de face) : taux de keypoints perdus par caméra et par phase.
2. Occultation des chevilles et des pieds par les disques et la barre, par caméra.
3. Distribution du bruit de chaque métrique sur reps propres ⇒ seuils.
4. Précision / rappel par faute ⇒ cibles §1.
5. Signal expérimental « raccourcissement épaule ↔ hanche » ⇒ activé ou abandonné.

## 9. Coaching vocal

- Menu : deadlift accepté (drapeau), sumo refusé poliment.
- **Agent d'apprentissage deadlift** (première séance) : module de textes séparé ; placement en 5 étapes (pieds largeur de hanches, barre au-dessus du milieu du pied, prise juste à l'extérieur des jambes, tibias à la barre, poitrine haute et mise en tension), puis 2–3 reps à la barre vide qui servent aussi à la calibration (§5.5).
- **Calibration caméra** : si aucun fichier de rig n'existe, la calibration actuelle demande 2 squats lents au poids du corps. On la garde telle quelle (elle calibre les caméras, pas l'exercice) et l'agent deadlift l'annonce.
- **Politique de cues deadlift** :
  - jamais de cue correctif pendant le tirage ;
  - cues de placement pendant `PLACEMENT` immobile, ou au sol entre deux reps ;
  - 1 cue correctif maximum par rep, priorité : perte de position (F2/F3/F1) > lockout (F4/F5) > asymétrie (F6) > bras (F7) ;
  - compte des reps au lockout ;
  - même délai minimal entre cues identiques que le squat.
- Le pipeline deadlift choisit lui-même la clé de cue et l'envoie par IPC : le `FAULT_TO_CUE_MAP` global (`coaching/cue_cache.py:87-94`) n'est ni utilisé ni modifié.
- Nouvelles clés audio pré-générées hors ligne (`generate_cue_audio.py`), lecture locale ; repli TTS existant si un fichier manque.
- Récap de série : diagnostic deadlift → LLM local, texte deadlift séparé.
- Enregistrement : `BiomechanicsRecorder(exercise="conventional_deadlift")` pour que les séances, reps et `CueEvent` soient étiquetés (base pour l'apprentissage futur, hors périmètre aujourd'hui).

## 10. Edge (Jetson)

- Nouveaux calculs (machine à états, géométrie, modèle de placement, diagnostic) : fermés, négligeables.
- Coût principal : détection de barre sur 3 vues. Mesure dédiée sur Jetson Orin Nano Super avec RTMPose + barre en même temps, TensorRT FP16. Leviers §6.7.
- Rien de nouveau dans la boucle conversationnelle ne dépend du cloud. L'annotation et l'entraînement se font hors ligne.

## 11. Jalons et critères d'acceptation

| Jalon | Contenu | Critère d'acceptation | Dépend de |
|---|---|---|---|
| **J0 — Filet squat** | Environnement de test (`requirements.lock`), golden master §3.4, manifeste de gel, test des alias, drapeau `NOWVA_ENABLE_DEADLIFT` (sans effet) | Golden master vert sur `main` ; un seuil squat modifié volontairement le fait échouer | — |
| **J1 — Connaissance + première capture** | `KNOWLEDGE.md` validé par Ambaka ; `deadlift/frame.py` + tests de signe ; modèle géométrique §5.4 + tests ; **première séance de capture brute** (round 1) | Revue signée ; tests verts ; vidéos et calibrations stockées ; keypoints de barre décidés (§6.2) | J0 |
| **J2 — Simulateur deadlift** | Générateur + scénarios + chemins corrigés | Chaque scénario produit sa vérité terrain | J1 |
| **J3 — Cœur deadlift (pose seule)** | Machine à états avec repli poignets, métriques, règles F1–F7, porte de départ, sol | Sur simulateur : 100 % des reps comptées, chaque faute injectée détectée, scénario propre sans faute | J2 |
| **J4 — Barre 3D** | Annotation, entraînement, export, tracker 3D, association multi-vues, sol | Rappel ≥ 95 % barre au sol ; erreur 3D ≤ 1,5 cm sur gabarit ; mesure de temps sur Jetson | J1 (données) |
| **J5 — Intégration** | `deadlift/process.py`, bifurcation `main.py`, voix, cues, enregistrement, derrière le drapeau | Séance complète sur le rack (dire « deadlift » → placement → reps comptées → cue → récap) ; golden master squat inchangé, drapeau éteint comme allumé | J0, J3, J4 |
| **J6 — Validation réelle** | Round 1 annoté → réglage des seuils ; round 2 pilote → métriques finales | Cibles §1 atteintes sur lifters jamais vus | J5 |
| **J7 — Diagnostic deadlift** | Graphe statique, score de rep, récap | Sur séries annotées : cause n°1 = annotation du coach dans ≥ 70 % des cas | J6 |
| **J8 — Démo** | Scénario héros : barre trop loin → « avance de 4 cm » → rep suivante corrigée ; plan B : hanches qui montent | 10 répétitions consécutives de la démo sans erreur ; budget Jetson respecté | J7 |

La capture réelle démarre dès J1, en parallèle du code, parce que la barre (J4) et les seuils (J6) en dépendent.

## 12. Risques et parades

| Risque | Parade |
|---|---|
| Toucher au squat par accident | Architecture séparée, drapeau, golden master, manifeste de gel, CI |
| Dos rond non mesurable ⇒ promesse impossible | Proxys + vocabulaire comportemental ; jamais de promesse de sécurité |
| Pose dégradée en position penchée | Mesure dès le round 1 ; si besoin, affiner RTMPose sur nos images (hors v1) |
| Occultation pieds/tibias par la barre et les disques | Mesure par caméra au round 1 ; prédiction Kalman ; seuils plus larges si source dégradée |
| Détecteur de barre qui ne généralise pas (disques, éclairage) | Données variées, découpage par lifter, négatifs (barre rangée) |
| Seuils inventés | Valeurs initiales seulement ; fixées sur données réelles avant la démo (accuracy d'abord) |
| Budget Jetson dépassé | Mesure à J4 ; leviers §6.7 |
| Duplication squat ↔ deadlift qui diverge | Accepté ; factorisation future uniquement sous golden master et avec accord |
| Barre lâchée / rebond des bumpers | Cas dans la machine à états et dans le simulateur |
| Confusion des repères (piège Y, Z) | `frame.py` unique + tests de signe |

## 13. Hors périmètre v1

Sumo, RDL et autres variantes ; apprentissage des probabilités ou du LLM ; correcteur de keypoints complet (fantôme animé) ; modèles ML (TCN) pour le dos rond ; VBT complet ; modification de tout fichier squat.

## 14. Questions restantes pour Ambaka

1. Qui annote les reps (Ambaka, un coach externe) ?
2. On compte le touch-and-go en v1, ou seulement les reps avec arrêt au sol ?
3. Où sont les poids actuels du détecteur de barre et sur quelles images ont-ils été entraînés ?
4. OK pour ajouter une CI GitHub Actions (§3.4.6) ?
5. OK pour la duplication assumée du moteur de diagnostic et de la boucle de session plutôt qu'une factorisation qui toucherait au squat ?
