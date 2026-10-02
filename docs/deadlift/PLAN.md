# Plan — Soulevé de terre conventionnel (deadlift) — v1

Statut : proposition, à valider par Ambaka. Version 4 du document.

## 0. Décisions déjà prises (ne pas rediscuter)

| Décision | Conséquence pour le plan |
|---|---|
| On sort du « squat only » de `CLAUDE.md` pour le deadlift | Chantier autorisé, mais il ne doit rien coûter au squat |
| **Le squat ne doit jamais être modifié et reste séparé du deadlift** | Aucun fichier squat n'est modifié ; le deadlift a son propre sous-processus et son propre package. Les fichiers partagés (voix, affichage) ne reçoivent que des ajouts derrière un drapeau. Aucun outil deadlift n'écrit dans un fichier d'état du squat. Deux filets de tests (pipeline et voix) prouvent que le squat ne bouge pas |
| Deadlift **conventionnel uniquement** | Sumo et autres variantes refusés **dans le code** |
| Dos rond : proxys indirects acceptés | On ne prétend jamais mesurer la colonne |
| Les caméras du rack voient la barre au sol | Détection et suivi 3D de la barre au sol à construire |
| Apprentissage du LLM : hors périmètre | On enregistre seulement les données, étiquetées par exercice |
| **Piège des axes** | §2 — règle bloquante |

## 1. Objectif produit et critères de succès

Boucle démo YC deadlift :

> « On fait du deadlift » → Nova guide le placement **en boucle fermée** (« avance un peu… encore… parfait ») → l'athlète tire → le système détecte une faute mesurable → **un** cue au bon moment (au sol, entre deux reps) → rep suivante corrigée → récap de série.

Fautes « héros » (démo, certifiées) : F1, F2, F3, F8. Les autres fautes v1 sont livrées en « provisoire » : cuées seulement à partir de « modéré », jusqu'à ce que les données suffisent (§8.3).

| Métrique (données réelles, lifters jamais vus) | Cible |
|---|---|
| Comptage des reps | ≥ 99 %, 0 rep fantôme sur séries propres |
| Événements (décollage, passage genoux, lockout, sol) | erreur médiane ≤ 100 ms |
| Précision par faute héros | ≥ 0,85 ; borne basse à 95 % (bootstrap par lifter) ≥ 0,75 ; aucun lifter de test < 0,70 ; ≥ 0,80 sur les séries naturelles |
| Rappel par faute héros | ≥ 0,70 ; borne basse à 95 % (bootstrap par lifter) ≥ 0,60 |
| Fausses corrections sur reps propres | ≤ 1 pour 10 reps (≥ 300 reps propres de test) |
| Erreur de mesure | ≤ 1/3 du seuil du niveau cué (§5.6) |
| Latence d'un cue | identique au squat (audio pré-enregistré) |
| Squat | golden masters pipeline **et** voix identiques |
| Edge | ≤ 33 ms/frame sur Jetson Orin Nano Super, ou mode dégradé défini (§10) |

## 2. Repères et piège des axes (règle bloquante)

### 2.1 Ce que dit le code

- Repère monde : mètres, **Y vers le bas** (Y plus grand = plus bas), X = gauche du sujet, **+Z = dos du sujet** (avant = −Z). Sources : `triangulation/person_calibration.py:5`, `triangulator.py:2`, `utils/foot_contact.py:32`, `kinematics/analytical_ik.py:449`.
- `.claude/rules/biomechanics.md` (« Larger Y = higher ») et `README.md:325` (« Z-forward ») sont **faux** par rapport au code.
- **L'axe Y du monde n'est pas la gravité.** `world_frame_from_standing` (`person_calibration.py:734-752`) prend Y = direction médiane hanche → cheville debout. Elle peut s'écarter de 2–4° de la verticale. Sur 55 cm de course de barre, 3° d'écart créent ≈ 2,9 cm de fausse dérive avant/arrière, soit le seuil de F3.

### 2.2 Repère du lifter

Aucune métrique deadlift n'utilise Y ou Z bruts. Tout passe par `src/biomechanics/deadlift/frame.py`.

**1. Vertical = gravité.** Ordre de préférence :

(a) **Mesure de gravité propre au deadlift.** Outil `scripts/tools/deadlift_gravity.py` (nouveau), avec la planche ChArUco posée à plat au sol :
- Pour **chaque caméra**, PnP mono-caméra de la planche, avec ses intrinsèques ; on en tire la normale de la planche dans le repère de cette caméra. La détection réutilise par import les fonctions de `triangulation/charuco.py`.
- Contrôle de cohérence : on ramène les normales dans le monde avec les rotations du rig actuel (lecture seule). Si elles divergent de plus de 0,5°, une caméra a bougé : l'outil refuse et demande une recalibration.
- **Seule écriture** : `~/.nowva/deadlift_gravity_<ids>.json`, qui contient la gravité de chaque caméra dans son propre repère.
- L'outil n'appelle jamais `calibrate_cameras.py extrinsics`. Cette commande sauvegarde `rig_calibration_*.json` (`calibrate_cameras.py:338-343`), ce qui renomme l'ancien fichier et rend orphelin le `_refined` utilisé par le squat (`pipeline_process.py:367-381`). Un test vérifie que l'outil ne touche à aucun fichier `rig_calibration_*`.
- Sur le rack produit, caméras fixes : une mesure en usine. Sur le rig de dev, à refaire après chaque déplacement des trépieds.

(b) **À l'exécution** : gravité monde = moyenne robuste, sur les caméras, de `R_caméra→monde · g_caméra`, avec les rotations de la calibration courante (y compris `_refined`, lecture seule). Une caméra qui s'écarte de plus de 1° des autres est exclue et signalée (elle a bougé). S'il reste moins de 2 caméras cohérentes, on passe au repli (c).

(c) **Repli** : axe hanche → cheville (comportement actuel), avec `gravity_source = "body"`. Les fautes sensibles à l'inclinaison (F2, F3, F5) ne sont alors cuées qu'à partir de « modéré ».

Option matérielle à discuter : un accéléromètre (~1 $) dans le boîtier donne la gravité en continu.

**2. Latéral** = axe de la barre au repos projeté sur le plan horizontal (ou ligne des hanches sans barre).

**3. Avant** = produit vectoriel, orienté talon → pointe.

**4. Conversions uniques**, nommées par leur sens : `height_m(point)` (le long de la gravité, vers le haut), `forward_of_m(point, reference)`, `lateral_of_m(point, reference)`, `sagittal_angle_deg(...)` (signé, positif vers l'avant). Interdiction de soustraire Y ou Z à la main ailleurs dans le package.

**5. Les métriques sont des différences** : barre vs sa hauteur de repos, poignets vs leur hauteur debout, barre vs milieu du pied. Elles ne dépendent pas d'une hauteur absolue du sol (§6.5).

**6. Tests obligatoires :**
- signe : barre qui monte pendant le tirage ; barre devant le pied ⇒ « avant » positif ;
- invariance : monde synthétique incliné de 3° et 5° avec gravité fournie ⇒ métriques identiques à ±2 mm et ±0,5° ;
- repli : l'erreur est rapportée et les seuils sont réellement élargis ;
- cohérence : une caméra « déplacée » de 2° dans le test est détectée et exclue.

**Observation pour Ambaka** (hors périmètre, on n'y touche pas) : le même biais d'axe affecte probablement des métriques squat comme l'inclinaison du tronc.

## 3. Isolation du squat (contrainte dure)

### 3.1 Architecture

```
main.py (~l. 452) : exercice = deadlift conventionnel ET NOWVA_ENABLE_DEADLIFT ?
   ├── non → pipeline_process.py (squat, inchangé, 0 ligne modifiée)
   └── oui → src/biomechanics/deadlift/process.py (nouveau sous-processus)
```

Le package `src/biomechanics/deadlift/` est construit **par composition**. Il n'hérite pas de `BiomechanicsPipeline`, qui chargerait le profil placeholder via `get_profile`, le moteur de règles et le compteur squat.

| Module | Rôle | Réutilise (import, lecture seule) |
|---|---|---|
| `process.py` | Boucle de session : calibration caméra, apprentissage, séries, repos, IPC dans les deux sens | `CameraCalibrationSession` ; helpers de `pipeline_process.py` importés s'ils sont au niveau module, sinon recopiés avec un commentaire d'origine |
| `pipeline.py` (`DeadliftPipeline`) | Par frame : pré-IK → IK → repère lifter → barre 3D → machine à états → métriques → règles | `build_preik_chain`, `AnalyticalIK`, `SegmentLengthEstimator` |
| `camera_provider.py` | Sous-classe de `MultiCameraPoseProvider` qui **ajoute** `get_frames_and_pose()` : 3 images synchronisées, vues 2D, squelette 3D | `MultiCameraPoseProvider`, `RTMPoseEstimator`, `DLTTriangulator` |
| `frame.py` | Repère lifter (§2.2) | fichiers de calibration et de gravité (lecture seule) |
| `bar_detector.py`, `bar_tracker_3d.py` | Barre multi-vues et 3D (§6) | `calibration.undistort_keypoints`, Ultralytics |
| `rep_counter.py`, `metrics.py`, `rules/` | Phases, métriques, fautes | — |
| `setup_model.py` | Placement attendu (§5.4) | — |
| `ipc.py` | Messages deadlift (§9.2) et politique de cues | client IPC existant |
| `diagnosis/` | Graphe statique (§7) | type `DiagnosisResult` (lecture seule) |
| `voice/` | Textes des agents deadlift | — |
| `config/deadlift.yaml` | Seuils et paramètres | `config/biomechanics.yaml` intact |

**Contrats d'interface à respecter** (testés à J3) :
- `CameraCalibrationSession` n'appelle pas le fournisseur, il appelle le **pipeline** (`pipeline_process.py:501-792`). `DeadliftPipeline` expose donc, avec la même sémantique, `process_frame`, `presence_only`, `last_frame`, `on_calibration_changed`, `body_calibration` et `_multi_camera_provider`. Un test fait tourner `CameraCalibrationSession` avec `DeadliftPipeline` sur des images rejouées.
- `get_frames_and_pose()` s'appuie sur l'état interne du fournisseur (`_capture.get_synced_frames()`, `_estimator.estimate_batch`, `_triangulator`, `_min_views`, `_primary_camera`). Elle appelle `_record_views()` comme `get_pose()` (`multi_camera.py:307-357`), pour garder les buffers de calibration et de dérive alimentés. Un test compare ses sorties à celles de `get_pose()` sur les mêmes images. Si la classe parente change, ce test et le golden master deadlift cassent : c'est voulu.

Le BiLSTM (entraîné sur squats, `pipeline.py:243-259, 815-854`) n'est jamais instancié côté deadlift. Le placeholder `profiles/deadlift.py` reste en place, jamais chargé quand le drapeau est actif.

On accepte la duplication (boucle de session, ≈ 150 lignes de calcul bayésien). Une factorisation future ne se fera qu'avec l'accord d'Ambaka et sous golden master.

### 3.2 Drapeau

`NOWVA_ENABLE_DEADLIFT` (défaut `false`). Drapeau éteint ⇒ tous les fichiers partagés se comportent exactement comme aujourd'hui, y compris les prompts et les cas d'erreur. Par exemple, « sumo deadlift » lance aujourd'hui le pipeline squat avec le profil placeholder : c'est conservé tel quel drapeau éteint.

### 3.3 Fichiers partagés modifiés : liste complète

Établie en suivant chaque message (dans les deux sens), chaque recherche de calibration, chaque écriture en base et l'écran. Tous les changements sont des **ajouts**, derrière le drapeau ou déclenchés uniquement par des messages que le squat n'émet jamais (champ `exercise`, types `dl_*`).

| Fichier | Changement |
|---|---|
| `src/main.py` ~452 | Choix du script de sous-processus |
| `src/main.py` ~386-430 (`_load_athlete_calibration`) | Branche deadlift : si la ligne `hip_hinge` n'a pas de mesures corporelles, lecture seule de la ligne `squat` comme point de départ |
| `src/main.py` ~989-1035 | Branche d'affichage pour les messages portant `exercise` ; transfert vers la voix (~1035) de `dl_cue`, et de `assessment_ready` **seulement s'il porte `exercise`** (le message squat sans ce champ reste non transféré, comme aujourd'hui) |
| `src/main.py` ~601-603 | Aucun changement : les messages voix → pipeline existants sont gérés par le processus deadlift (§9.2) |
| `src/visual/display.html` ~963, 1223, 1317, 1422 | Tuiles et libellés deadlift si `exercise` est présent ; tuiles squat inchangées |
| `src/agent/agents/prompts/main_menu_prompt.py:26` | Texte « squats et deadlift conventionnel » si drapeau |
| `src/agent/agents/main_menu_agent.py` | Si drapeau : `start_quick_exercise` refuse les exercices non pris en charge ; branche deadlift qui reconnaît le nom brut avec `deadlift/voice/names.py` (« deadlift », « conventional deadlift », « Barbell Conventional Deadlift » de la bibliothèque de programmes) et passe le nom canonique `"Barbell Deadlift"` à `check_calibration` / `start_calibration_mode` ; test de provenance de la calibration (§3.3, motif) ; texte de progression (~640) selon l'exercice |
| `src/agent/agents/quickExerciseAgent.py` | `:211` passe `exercise=` au `TeachingAgent` ; branche deadlift à la décision de calibration (~:175-190) qui applique le test de provenance. Les appels de `:112` et `:186` reçoivent le nom canonique fourni par la branche deadlift de `main_menu_agent` |
| `src/agent/agents/teaching_agent.py`, `calibration_agent.py`, `workout_agent.py` | Retour anticipé vers `deadlift/voice/` si exercice = deadlift ; code squat ni modifié ni réindenté |
| `src/agent/services/coaching_service.py` | Branche si `exercise` présent dans `rep_complete` / `diagnosis_complete` / `calibration_complete` / `assessment_ready` ; nouveau gestionnaire `dl_cue` ; `BiomechanicsRecorder(exercise="conventional_deadlift")` (~129, déjà supporté par `db/biomechanics_persistence.py:290`) ; requêtes de progression appelées avec `exercise="conventional_deadlift"` dans la branche deadlift (~159-160 ; les fonctions acceptent déjà ce paramètre) ; prompts deadlift |
| `src/agent/services/coaching_orchestrator.py` | Nouvelle méthode publique `play_cached_cue(cue_key, priority, max_age_s, data)` qui place un événement `cached_cue` dans la file existante (`_dispatch_cached_cue`, ~961-1021 : lecture via `session.say(allow_interruptions=False)`, accusé de livraison automatique si `data.fault_type` est présent). Ajout, jamais appelée par le squat. Branches de récap deadlift (~1227-1251, 1388-1390, 1407, 1510) |
| `src/agent/services/coaching_constants.py` | Clés `dl_*` ajoutées **après** la génération de leur audio. Sinon `_validate_expected_cues` (`audio_cue_service.py:260-272`) journaliserait des avertissements pendant les séances squat |
| `src/agent/services/progress_context.py` | Libellés et filtre par exercice |
| `scripts/tools/generate_cue_audio.py` | Prompts audio `dl_*` ajoutés |

**Jamais modifiés** : `pipeline.py`, `pipeline_process.py`, `profiles/*`, `diagnosis/*`, `faults/*`, `coaching/{cue_cache,ipc_bridge,session_tracker}.py`, `calibration.py`, `utils/foot_contact.py`, `pose/multi_camera.py`, `barbell_tracking/*`, `config/biomechanics.yaml`, `db/biomechanics_persistence.py`, `scripts/tools/calibrate_cameras.py`, `agent/agents/shared/helpers.py` (les noms deadlift sont reconnus par `deadlift/voice/names.py`, appelé seulement dans les branches deadlift : drapeau éteint, « conventional deadlift » se comporte exactement comme aujourd'hui).

**Motif de mouvement** : le deadlift utilise la clé existante `hip_hinge`, déjà associée à `"Barbell Deadlift"` (`calibration.py:28`). Les recherches de calibration existantes (`check_calibration`, `start_calibration_mode`, `_load_athlete_calibration`) marchent donc sans modifier `calibration.py`. Le RDL et le sumo partagent cette clé mais sont refusés en v1. S'ils sont ajoutés un jour, il faudra une clé distincte et une migration.

**Provenance de la calibration `hip_hinge`.** Aujourd'hui, drapeau éteint, « deadlift », « RDL » ou un programme « Barbell Conventional Deadlift » lancent le pipeline squat avec le profil placeholder. En mode calibration, celui-ci écrit sous `hip_hinge` un profil **de forme squat** (`pipeline_process.py:1073, 1499`). Le deadlift ne doit donc pas faire confiance à une ligne `hip_hinge` existante :
- chaque calibration écrite par le deadlift porte `thresholds.schema = "dl_v1"` ;
- une ligne sans ce marqueur est traitée comme **« pas de calibration deadlift »** : l'apprentissage et la mesure du bruit ont lieu. Seules ses mesures corporelles (`athlete_params`) servent de point de départ ; ses seuils et pics sont ignorés ;
- la fonction `deadlift/voice/calibration.py::is_deadlift_calibration(row)` est appelée par les branches deadlift de `main_menu_agent`, `quickExerciseAgent` et `main._load_athlete_calibration` ;
- tests : une ligne `hip_hinge` de forme squat ⇒ parcours « première fois » ; une ligne `dl_v1` ⇒ parcours « utilisateur connu ».

**Isolation des données.** Les lectures utilisées par le squat filtrent déjà sur `exercise="squat"` (`get_last_completed_session`, `get_progress_baseline`, `get_score_progress`, `get_multi_session_fault_trends` dans `db/biomechanics_persistence.py:738-830, 922+`). Les séances deadlift, enregistrées avec `exercise="conventional_deadlift"`, n'entrent donc jamais dans le contexte squat. Les fonctions non filtrées (`get_fault_progress`, `get_rep_kinematics_history`, `get_cue_effectiveness`) n'ont aucun appelant aujourd'hui ; tout futur appelant devra filtrer par exercice. `rep_kinematic_summary` deadlift porte `dl_schema: 1` et `exercise`.

**Préfixe `dl_`** sur tous les types de faute et toutes les clés de cue deadlift, pour éviter les collisions avec `PREEMPTIVE_TEXT["chest_up"]`, avec `END_OF_REP_FAULT_TYPES` (qui contient déjà `lockout`) et avec les requêtes non filtrées (`get_fault_progress`, `get_cue_effectiveness`). Les types `dl_*` ne sont **pas** ajoutés à `END_OF_REP_FAULT_TYPES` : ils sont émis avec le numéro de la rep courante avant `rep_complete`, donc la liaison cue ↔ rep et l'évaluation « rep suivante corrigée » (`biomechanics_persistence.py:467-489`) fonctionnent normalement.

### 3.4 Filet de sécurité squat (Jalon 0, avant toute ligne deadlift)

**1. Golden master pipeline squat** — `tests/test_biomechanics/test_squat_golden.py` :
- Scénarios rejoués via le harnais multi-caméra de `tests/test_biomechanics/test_pipeline.py:329-392` (`_FakeProvider`, `_FakeClock`, `world_squat_points`, `squat_depth_profile`) : série propre, valgus, inclinaison, `go_deeper`, perte de détection, asymétrie. Mono-caméra via `estimate_both` simulé ; BiLSTM via un faux déterministe.
- Par frame : `model_dump(mode="json")` sans `latency_ms`, NaN → null. Puis `SessionTracker` + `IPCBridge` (client simulé) et `force_end_set` : capture de tous les messages IPC, y compris `diagnosis_complete`.
- Déterminisme : `time.time` remplacé dans `session_tracker` et `ipc_bridge`, `random` initialisé.
- Comparaison par sérialisation canonique (flottants arrondis à 1e-9), dans l'environnement figé de `requirements.lock`. Régénération seulement avec une variable d'environnement explicite.
- Drapeau éteint **et** allumé.

**2. Golden master voix squat** — `tests/test_coaching_squat_golden.py` (même niveau que `tests/test_coaching_service.py`, qui construit déjà `CoachingService(session=None, state=None)`) :
- Le flux IPC capturé au point 1 est rejoué dans `CoachingService` + `CoachingOrchestrator`, avec TTS, LLM et base simulés.
- On capture la séquence des cues joués, le texte exact des prompts de récap et les opérations de l'enregistreur (`_ops_*`, fonctions pures).
- `time.monotonic` remplacé dans l'orchestrateur ; `random` initialisé (choix des variantes audio, `audio_cue_service.py:344`) ; ordre des tâches asyncio maîtrisé (on attend que la file soit vide entre deux messages).
- Drapeau éteint et allumé. C'est le filet qui couvre les fichiers voix du §3.3.

**3. Golden master des prompts** : texte des agents squat (menu drapeau éteint, apprentissage, calibration, séance).

**4. Test des alias** : alias squat → `SquatProfile` et motif `squat` ; avec le drapeau, « sumo deadlift » est refusé et ne lance aucun pipeline.

**5. Manifeste de gel** : `tests/test_squat_freeze.py` vérifie l'empreinte SHA-256 de la liste « Jamais modifiés ». Le mettre à jour exige l'accord d'Ambaka.

**6. Test « aucune écriture squat »** : les outils et le processus deadlift (gravité, enregistreur, séance) tournent dans un `HOME` temporaire ; on vérifie qu'aucun fichier `rig_calibration_*`, `intrinsics_*` ou état squat n'est créé ou modifié. Variante base de données : avec une base qui contient des séances squat **et** deadlift, le contexte squat (accueil, progression, récap) est identique à celui d'une base qui ne contient que les séances squat.

**7. Rejeu de vraies séances squat** : impossible aujourd'hui (`user_test_runs/` ne contient que des sorties). Dès que l'enregistreur existe (J1), on enregistre 3 à 5 séances squat brutes et on les rejoue dans le pipeline squat inchangé, via un fournisseur de rejeu injecté dans le test. Snapshot des sorties.

**8. CI** : aucune aujourd'hui. Proposition : GitHub Actions avec `requirements.lock`, `PYTHONPATH=src pytest tests/ -x` sur chaque PR (à valider).

**9. Golden master deadlift** (J3) : protège le deadlift contre de futurs changements côté squat dans le code partagé (pré-IK, IK, triangulation, fournisseur caméra).

Preuve que le filet fonctionne : modifier volontairement un seuil squat ou un texte de cue squat fait échouer le bon golden master.

## 4. Le mouvement : phases et machine à états

### 4.1 Phases

`APPROCHE` (debout) → `POSITION` (debout au-dessus de la barre, pieds plantés) → `PLACEMENT` (penché, mains sur la barre, immobile) → `TIRAGE` (décollage → passage des genoux → fin du tirage) → `HAUT` → `DESCENTE` → `AU SOL`.

Deux arrêts servent à corriger : `POSITION` (pieds) et `PLACEMENT` (hanches, épaules). Pendant le tirage (≈ 1 s), aucun cue correctif.

### 4.2 Références personnelles prises à l'approche

Pendant `APPROCHE`, debout immobile (≥ 1 s), on enregistre :
- angles sagittaux **signés** de hanche, genou, tronc et coude debout ;
- hauteur des poignets bras pendants. C'est la **hauteur de lockout attendue** : directement en mode poignets, et moins le décalage poignet → barre en mode barre ;
- ancres des pieds et milieu du pied (§6.5).

F4, F5 et F7 sont mesurés par rapport à ces références, ce qui retire les biais de keypoints propres à chaque personne.

### 4.3 Signal de rep

1. Hauteur du centre de barre (barre 3D, §6), relative à sa hauteur de repos.
2. Repli : hauteur du milieu des poignets, relative à sa hauteur en `PLACEMENT`. La référence de lockout vient de la position debout, pas de la barre, donc pas de circularité.
3. Jamais l'inversion du signal de hanche du placeholder.

### 4.4 Machine à états (`deadlift/rep_counter.py`)

Le comptage est **piloté par la barre** ; la posture est jugée séparément par les fautes.

| Transition | Condition (valeurs initiales, `config/deadlift.yaml`) |
|---|---|
| `APPROCHE → POSITION` | debout, milieu du pied à moins de 15 cm de la barre (horizontalement), pieds immobiles ≥ 0,5 s |
| `POSITION → PLACEMENT` | mains à moins de 10 cm au-dessus de la barre, immobiles ≥ 0,3 s |
| `PLACEMENT → POSITION` / `APPROCHE` | se relève sans soulever |
| `PLACEMENT → TIRAGE` | barre > repos + 3 cm et vitesse verticale > 0,10 m/s |
| `TIRAGE → HAUT` | barre ≥ hauteur de lockout attendue − 8 cm, vitesse verticale \|v\| < 0,05 m/s pendant ≥ 3 frames, tronc à moins de 35° de la verticale (garde-fou large) |
| `TIRAGE → AU SOL` sans `HAUT` | **rep ratée** (la barre redescend sans avoir atteint la hauteur de lockout − 8 cm) : événement, non comptée |
| `HAUT → DESCENTE` | vitesse < −0,10 m/s |
| `DESCENTE → AU SOL` | barre ≤ repos + 3 cm → **rep comptée et analysée** |
| `DESCENTE → TIRAGE` | touch-and-go : la vitesse redevient positive à moins de 5 cm du sol |
| `AU SOL → APPROCHE` | se relève sans la barre |

- Le seuil de 8 cm est choisi pour qu'un lockout mou (hanche fléchie de 20° ⇒ barre ≈ 3–5 cm plus basse) ou en hyperextension soit **compté** puis signalé par F4/F5. Une rep bloquée à hauteur des genoux n'est pas comptée. La valeur est validée sur les reps annotées (§8).
- **Comptage au sol** : un seul message `rep_complete` (§9.2). En touch-and-go, le compte tombe au contact du sol.
- Barre lâchée depuis le haut : rep comptée, **pas de faute** (normal avec des bumpers), simple information.
- Ajustements de la barre au sol (< 3 cm) : ignorés. Fin de série : `AU SOL`/`APPROCHE` pendant `set_timeout_seconds` (30 s) ou « j'ai fini ».
- Hauteur de repos : médiane pendant `POSITION`/`PLACEMENT`, par série.
- Touch-and-go : aucun cue entre les reps ; les corrections passent au repos.

## 5. Fautes, mesures, seuils

Mesures dans le repère du lifter (§2.2). Seuils initiaux dans `config/deadlift.yaml`, fixés définitivement sur données réelles (§8). Trois niveaux : léger / modéré / sévère. Le niveau minimal cué dépend du budget d'erreur (§5.6) et du statut héros / provisoire.

### 5.1 Fautes v1

| # | Faute | Phase | Mesure | Seuils initiaux | Cue |
|---|---|---|---|---|---|
| F1 ★ | **Barre pas au-dessus du milieu du pied** | `POSITION` (debout, pieds plantés), recontrôle en `PLACEMENT` | `bar_forward_of_midfoot_cm` | 3 / 5 / 8 cm | boucle fermée : « avance un peu / encore » → « parfait » |
| F8 ★ | **Hanches trop basses / trop hautes** | `PLACEMENT` | Hauteur de hanche vs bande du modèle (§5.4) | hors bande de 4 / 7 / 10 cm | « Monte / descends un peu les hanches » |
| F9 | **Épaules derrière la barre** | `PLACEMENT` | Articulation de l'épaule vs barre, axe avant | derrière de 2 / 4 / 6 cm | « Épaules au-dessus de la barre » |
| F2 ★ | **Hanches qui montent avant les épaules** | Décollage → passage des genoux | Variation mesurée de l'angle du tronc **moins** la variation prédite par le modèle (§5.4) ; contrôle croisé : rapport des montées hanche / épaule | 10° / 15° / 20° ; rapport > 1,4 | « Poitrine et hanches ensemble — pousse le sol » |
| F3 ★ | **Barre qui s'éloigne du corps** | Tirage | Écart avant max du centre de barre vs sa position au décollage | 3 / 5 / 8 cm | « Barre collée aux jambes » |
| F4 | **Lockout incomplet** | `HAUT` (frame la plus haute) | Déficit signé d'extension de hanche ou de genou vs référence debout (flexion résiduelle) | 8° / 12° / 20° | « Serre les fessiers, finis debout » |
| F5 | **Hyperextension au lockout** | `HAUT` | Tronc en arrière, signé, vs référence debout | 8° / 12° / 18° | « Grandis-toi, ne te penche pas en arrière » |
| F6 | **Asymétrie** | Tirage | Décalage latéral du bassin vs milieu des pieds ; inclinaison de barre (différence de hauteur des extrémités) | 3 / 5 / 7 cm | « Pousse pareil dans les deux pieds » |
| F7 | **Bras pliés** | Tirage | Flexion du coude vs référence debout | 15° / 25° / 35° | « Bras longs et tendus » |

★ = faute héros. F4 et F5 sont indépendantes du comptage : un lockout mou ou en hyperextension est compté, puis signalé.

La rotation du tronc est retirée de F6 en v1, car une prise mixte la déclenche à tort. Le type de prise est demandé et enregistré.

### 5.2 Fautes v1.1

- arrachage sans mise en tension (pic d'accélération au décollage) ;
- descente non contrôlée ;
- genoux qui avancent avant que la barre passe les genoux à la descente ;
- « hitching » (genou qui se replie au lockout) ;
- position de la tête ;
- largeur de prise et de pieds ;
- genoux qui rentrent ;
- talons qui décollent ;
- perte de vitesse de barre (fatigue, base de « charge trop lourde » et du VBT).

### 5.3 Dos rond : proxys honnêtes

- Les 21 keypoints (COCO 17 + pointes + talons) n'ont aucun point sur la colonne : on **ne mesure pas** le dos rond et on ne l'affirme jamais.
- « Perte de position » = F2 + F3 + F9.
- **Signal expérimental** : raccourcissement de la distance épaule ↔ hanche vs la longueur debout.
  - Bruit : l'écart-type d'un keypoint triangulé est de ≈ 1,6 cm au seuil de confiance 0,6 et ≈ 2,4 cm à 0,4 (`utils/segment_lengths.py:19-21`). C'est comparable à l'effet attendu (1–3 cm).
  - Le signal est donc moyenné sur la phase de tirage.
  - Il n'est activé que si les données montrent une séparation nette (AUC ≥ 0,8 sur lifters jamais vus) entre reps « dos rond » annotées et reps propres.
- Cues comportementaux uniquement (« dos plat, gaine-toi »), jamais médicaux ni de « sécurité ».
- La `BackRoundingRule` actuelle n'est pas utilisée : elle mesure la variation d'inclinaison du tronc, qui est le mouvement normal du deadlift.

### 5.4 Modèle du placement attendu (cœur de l'IP)

Équivalent deadlift de `expected_trunk_lean_geometric` : la bonne position pour **ce corps-là**.

**Contraintes**, sous forme de bandes :
- barre au-dessus du milieu du pied ;
- articulation de l'épaule 0–6 cm devant la barre (omoplates au-dessus de la barre) ;
- bras verticaux ;
- tibia en contact avec la barre.

**Entrées :**
- **longueurs projetées dans le plan sagittal** du tibia et du fémur, mesurées directement sur les frames de `PLACEMENT` (genoux ouverts et pieds tournés raccourcissent la projection), et non leur longueur 3D complète ;
- torse et pied (`SegmentLengthEstimator`) ;
- **longueur des bras** : estimateur propre au deadlift (épaule → coude → poignet) ; `SegmentLengthEstimator` ne mesure pas les bras et n'est pas modifié ;
- décalage poignet → prise : 6–9 cm, appris par personne quand barre et poignets sont visibles ensemble ;
- distance axe de barre ↔ ligne cheville–genou au contact : rayon de barre (1,4 cm) + tissu devant le tibia (3–5 cm), 5 cm au départ ;
- position de la barre et de la cheville, mesurées. Le modèle travaille en relatif à la cheville et n'a pas besoin du sol.

**Résolution 1 — départ :**
- On place l'épaule dans sa bande, au-dessus de la barre (hauteur = barre + prise + bras), et on fixe la cheville.
- La chaîne cheville → genou → hanche → épaule n'a plus qu'un degré de liberté.
- La contrainte de contact du tibia a **deux solutions** : on garde celle avec le genou devant la cheville et une flexion du genou dans [40°, 120°].
- S'il n'y a aucune solution, la morphologie est hors modèle : F8 et F9 sont désactivées pour cette personne et c'est journalisé.

**Résolution 2 — passage des genoux** : barre à hauteur des genoux, tibias quasi verticaux, épaules au-dessus de la barre, bras verticaux. Elle donne l'angle du tronc prédit au passage des genoux. **La variation prédite pour F2** est l'angle du tronc prédit au passage des genoux moins l'angle prédit au départ.

**Sorties** : bande de hauteur de hanche, angles du tronc prédits (départ, passage des genoux), flexion du genou attendue.

**Validation et critère d'usage :**
- tests sur géométrie connue ;
- cohérence : fémurs longs ou bras courts ⇒ dos plus horizontal ;
- comparaison avec les placements annotés « bons » par le coach : la référence est la hauteur de hanche **mesurée** sur ces placements (marqueur sur le grand trochanter vu par la caméra sagittale de référence, erreur ≤ 1 cm, §8.4) ; l'erreur du modèle = hauteur prédite − hauteur mesurée. **F8 n'est cuée au niveau modéré que si cette erreur est ≤ 2,3 cm au 95e percentile** (1/3 du seuil modéré de 7 cm). Si l'erreur est ≤ 3,3 cm, F8 est cuée seulement au niveau sévère. Au-delà, F8 est désactivée et le plan B de la démo s'applique (§11, J8).

### 5.5 Personnalisation

- F4, F5, F7 : relatives à la référence debout de la séance.
- F1, F3, F6 : distances absolues.
- F2, F8, F9 : relatives au modèle personnalisé.
- Une série d'échauffement mesure le bruit de chaque métrique : seuil = max(seuil de base, 3 × bruit), plafonné par une limite qu'aucune calibration ne dépasse. Jamais de « pic observé + marge » (comme le squat), qui peut normaliser une faute.
- Stockage sous `movement_pattern = "hip_hinge"` (§3.3). Le chemin deadlift ne passe jamais par le repli `or "squat"` de `pipeline_process.py:1073`.
- Les mesures corporelles stockées pour le squat sont lues comme point de départ (lecture seule, via la branche deadlift de `_load_athlete_calibration`), puis remesurées dans la séance.

### 5.6 Budget d'erreur

Règle : on ne cue un niveau que si l'erreur de mesure au 95e percentile, mesurée contre la vérité terrain (§8.4), est ≤ 1/3 du seuil de ce niveau. Sinon, on cue à partir du niveau supérieur.

| Faute | Seuil léger | Erreur max pour « léger » | Erreur attendue (à vérifier) | Niveau cué au départ |
|---|---|---|---|---|
| F1 | 3 cm | 1 cm | ≤ 1 cm (barre statique + ancres moyennées sur ≥ 15 frames) | léger |
| F3 | 3 cm | 1 cm | 1–2 cm (dynamique) | modéré tant que non prouvé |
| F6 | 3 cm | 1 cm | 1–2 cm | modéré |
| F2 | 10° | 3,3° | 2–4° + erreur du modèle | modéré tant que non prouvé |
| F4 / F5 | 8° | 2,7° | 2–3° (relatif) | modéré tant que la vérité terrain n'est pas assez précise (§8.4) |
| F7 | 15° | 5° | 3–5° | léger |
| F8 | 4 cm | 1,3 cm | voir §5.4 | modéré ou sévère selon §5.4 |
| F9 | 2 cm | 0,7 cm | 1–1,5 cm | modéré |

## 6. Barre au sol : détection et suivi 3D

### 6.1 Existant

- `BarbellDetector` (`barbell_tracking/detector.py`) : YOLO11n-pose, ne renvoie que la meilleure détection (`:113`).
- Les poids (`models/barbell_keypoints.pt`) ne sont pas dans le repo ; on ignore leurs données d'entraînement.
- Désactivé par défaut (`config.py:145`). En multi-caméra, il ne tourne que sur la caméra principale, en 2D (`pipeline.py:613-624` ; `multi_camera.py:307-357` ne renvoie que l'image principale).
- Les extrémités de la barre ne sont triangulées que pendant la calibration caméra, pour l'échelle, puis jetées (`person_calibration.py:599-601, 852, 968`).
- `BarPathRule` mélange les axes : non réutilisée.

### 6.2 Keypoints

Barre chargée : 2 keypoints = **centre de la face extérieure du disque extérieur**, à gauche et à droite (sur l'axe de la barre). C'est grand, contrasté et visible de face comme à 45°. Barre vide : extrémités du manchon, en classe distincte. On confirme ce choix sur les premières images réelles (J1) avant l'annotation en masse.

### 6.3 Données et entraînement

1. Récupérer les poids actuels chez Ambaka et les évaluer tels quels (J1). Décision : affiner ce modèle ou repartir de YOLO11n-pose.
2. Images extraites des enregistrements bruts (§8.2) : barre au sol, en tirage, en haut, barre rangée dans les J-hooks (distracteur), barre vide, bumpers, fonte, couleurs, éclairages, tenues, lifters. Les 3 caméras.
3. Annotation dans **CVAT auto-hébergé**, aussi utilisé pour les étiquettes de reps : 2 000–3 000 images équilibrées par caméra et par phase, avec des négatifs.
4. Entraînement Ultralytics (déjà en dépendance), export ONNX → TensorRT FP16.
5. Évaluation par lifter : rappel ≥ 95 % sur barre au sol ; erreur keypoint (px) ; faux positifs sur barre rangée ≤ 1 %.

### 6.4 Exécution en série

1. `camera_provider.get_frames_and_pose()` fournit les 3 images (§3.1).
2. `bar_detector.py` appelle le modèle Ultralytics directement, **en lot sur les 3 vues**, et renvoie **toutes** les détections au-dessus du seuil. Nouvelle classe ; `BarbellDetector` n'est pas modifié.
3. Correction de distorsion des keypoints (`calibration.undistort_keypoints`).
4. Association multi-vues : combinaison avec la plus petite erreur de reprojection, compatible avec les mains et les pieds du lifter. Cela élimine la barre rangée.
5. Gauche/droite via la ligne des épaules (même principe que `_label_bar_ends_by_shoulders`, recopié).
6. DLT sur les 2 extrémités (≈ 40 lignes dans le package ; les fonctions de `triangulator.py` sont privées). Rejet si l'erreur de reprojection dépasse le seuil.
7. Kalman 3D à vitesse constante par extrémité, prédiction pendant les trous courts.
8. Sortie `BarState3D` (repère lifter) : centre, hauteur relative au repos, avant/arrière vs milieu du pied, inclinaison (cm et °), vitesse verticale (m/s), `source` (`bar` / `wrist_proxy`), confiance.

### 6.5 Milieu du pied et sol

`foot_contact.py` ne publie pas ses ancres et reste intouché. Le package deadlift fait son propre suivi :
- Les ancres de pied (talons, pointes) sont verrouillées dans le repère lifter pendant `APPROCHE`/`POSITION` (pieds immobiles, moyenne sur ≥ 15 frames).
- **Milieu du pied** = milieu talon ↔ pointe, **verrouillé avant le tirage**. Il ne dépend donc pas des occultations par les disques pendant le tirage.
- Les métriques sont relatives (§2.2.5) ; le sol absolu ne sert qu'à l'affichage et aux contrôles.
- Sol absolu : barre au repos − rayon du disque si le diamètre est connu (métadonnée, 45 cm pour bumpers et disques de compétition). Sinon, ancres de pied − décalage keypoint → sol. Ce décalage dépend de la personne et des chaussures : il est estimé par séance quand la barre au repos est disponible, avec des valeurs par défaut sinon.
- Remise à zéro quand la calibration caméra change.

### 6.6 Repli sans barre

Milieu des poignets moins le décalage appris. F1, F3 et l'inclinaison de F6 passent en `source = wrist_proxy` : seuils élargis, cue au niveau « modéré » minimum. Si la confiance est trop basse, la faute n'est pas émise.

### 6.7 Coût edge

Mesuré sur Jetson **dès J1** (§10). Leviers dans l'ordre :
1. TensorRT FP16 ;
2. entrée 480 px ;
3. détection 1 frame sur 2 avec prédiction Kalman entre les deux ;
4. recadrage autour des mains et des pieds ;
5. seulement les 2 caméras à 45°.

## 7. Diagnostic après la série (graphe statique)

### 7.1 Principe

Même méthode que le squat : symptômes → causes candidates → probabilité de départ écrite à la main × test de preuve → normalisation avec fuite → noisy-OR. Sortie au **même schéma `DiagnosisResult`**.

Module séparé `deadlift/diagnosis/` (`symptoms.yaml`, `causes.yaml`, `evidence_tests.py`, `parameter_deltas.py`, `engine.py`). Le moteur squat charge ses graphes en globales à l'import (`diagnosis/graph/loader.py:114-116`), donc il n'est pas réutilisable sans modification.

Sorties simples pour un petit LLM local : `cause_id`, niveau, score, **un** delta chiffré, une phrase.

### 7.2 Symptômes v1

`dl_bar_not_over_midfoot` (F1), `dl_setup_hip_height` (F8), `dl_shoulders_behind_bar` (F9), `dl_hips_rise_early` (F2), `dl_bar_drift` (F3), `dl_incomplete_lockout` (F4), `dl_overextension` (F5), `dl_lateral_shift` (F6), `dl_bent_arms` (F7).

### 7.3 Causes v1 (probabilités écrites à la main, documentées)

| Cause | Niveau | Delta | Impliquée par |
|---|---|---|---|
| `dl_feet_position` | 1 | « avance de X cm » | F1, F3 |
| `dl_hips_too_low` | 1 | « hanches X cm plus haut » | F8, F2 |
| `dl_hips_too_high` | 1 | « hanches X cm plus bas » | F8, F3 |
| `dl_shoulders_position` | 1 | — | F9, F3 |
| `dl_lats_not_engaged` | 1 | — | F3 |
| `dl_slack_not_pulled` | 1 | — | F2 |
| `dl_glutes_not_finishing` | 1 | — | F4 |
| `dl_overextension_habit` | 1 | — | F5 |
| `dl_uneven_stance_or_grip` | 1 | « décale ta prise de X cm » | F6 |
| `dl_weight_too_heavy` | 2 | « baisse d'environ 10 % » | dégradation au fil de la série |
| `dl_weak_off_floor` | 3 | — | F2 malgré un bon placement |
| `dl_hip_hamstring_mobility` | 3 | — | F8 « trop bas » qui ne se corrige pas |
| `dl_unilateral_weakness` | 3 | — | F6 persistant |
| `dl_anthropometric_context` | 0 | — | « fémurs longs / bras courts : dos plus horizontal attendu » |

### 7.4 Score de rep

5 dimensions dans [0, 1], poids statiques :
- placement (F1, F8, F9) : 25 %
- coordination (F2) : 25 %
- trajectoire de barre (F3) : 20 %
- lockout (F4, F5) : 15 %
- symétrie (F6) : 15 %

### 7.5 Fantôme

Le correcteur de keypoints du squat n'est pas réutilisé. Option démo : le placement idéal du §5.4 affiché comme fantôme statique.

## 8. Données

### 8.1 Connaissance statique

`docs/deadlift/KNOWLEDGE.md` contient :
- les phases ;
- la définition de chaque métrique (repère, signe, phase) ;
- les seuils initiaux et leur justification ;
- les cues ;
- le modèle géométrique.

Sources : modèle de placement classique du coaching, manuels de référence en préparation physique. Revue par Ambaka et idéalement par un coach de force externe. Chaque seuil porte « à valider sur données ».

### 8.2 Enregistreur brut et rejeu (J1, prérequis)

**`scripts/tools/record_rig.py`** : outil autonome qui ouvre les 3 caméras via `multi_capture` (import) et écrit :
- une vidéo par caméra (H.264 haute qualité ou MJPEG) ;
- l'horodatage de chaque frame ;
- une **copie** des fichiers de calibration et de gravité ;
- les métadonnées : lifter, charge, diamètre des disques, prise, chaussures, ceinture.

Il ne passe par aucun pipeline et n'écrit rien dans `~/.nowva`.

**`deadlift/replay_provider.py`** : relit un enregistrement avec la même interface que le fournisseur caméra. Il sert au pipeline deadlift hors ligne et, injecté dans un test, au rejeu des séances squat dans le pipeline squat inchangé (§3.4.7).

Volume : ≈ 5–10 Mo/s, soit ≈ 5 Go pour 10 min. Stockage local, sauvegarde chiffrée.

### 8.3 Données réelles et plan de test

**Round 1 — équipe, semaines 1–2 :**
- 2–3 personnes, barre vide et charges légères ;
- reps propres et fautes volontaires sans danger (F1–F9) ;
- dos rond montré uniquement avec barre vide ou bâton.

**Round 2 — pilote :**
- 10–15 lifters de 1,55 à 1,95 m, aux ratios fémur/torse/bras variés, débutants à confirmés ;
- séries naturelles (RPE ≤ 8) et **séries de fautes scénarisées**.

**Jeu de test** : **≥ 8 lifters** jamais vus à l'entraînement ni au réglage.
- **Fautes héros (F1, F2, F3, F8) : ≥ 100 positifs chacune.** Avec un rappel de 0,7, cela donne ≈ 82 cues émis ; à une précision de 0,85, la borne basse de Wilson est ≈ 0,76 et celle du rappel ≈ 0,60. C'est un plancher de dimensionnement.
- **Fautes provisoires (F4–F7, F9) : ≥ 40 positifs chacune.** Intervalles rapportés tels quels ; cues seulement à partir de « modéré » jusqu'à 100 positifs.
- **≥ 300 reps propres** pour le taux de fausses corrections.
- Les reps d'un même lifter ne sont pas indépendantes. **La porte de J6 utilise un bootstrap par lifter** (on rééchantillonne des lifters, pas des reps). Elle exige en plus une précision minimale par lifter (≥ 0,70) pour les fautes héros.
- **Séries scénarisées et naturelles rapportées séparément.** Les fautes scénarisées à charge légère surestiment la précision réelle. Les séries naturelles sont filmées en mode observation (cues coupés), et une faute héros doit atteindre ≥ 0,80 de précision sur ≥ 30 cas naturels ; sinon elle reste provisoire.
- Volume : ≈ 25 séries de 5 reps par lifter de test, en 2 séances, soit 8 × 25 × 5 = 1 000 reps, surtout à charge légère.

**Annotation (CVAT, vidéo des 3 vues) :**
- par rep : chaque faute (présente / niveau) ;
- sur un sous-ensemble, horodatage du décollage, du passage des genoux, du haut et du sol, pour évaluer la machine à états seule ;
- 2 annotateurs sur 20 % des reps, accord cible κ ≥ 0,6 par faute ;
- désaccords tranchés par Ambaka ;
- une faute avec κ < 0,6 est redéfinie avant d'être cuée.

**Volumes d'entraînement et de réglage** : ≥ 300 reps annotées et ≥ 50 positifs par faute, en plus du jeu de test ; 2 000–3 000 images de barre.

**Consentement** écrit, anonymisation, stockage local.

### 8.4 Vérité terrain

| Grandeur | Méthode | Précision visée |
|---|---|---|
| Barre statique / F1 | Scotch au sol + gabarit de pied : barre à 0 / 3 / 6 / 10 cm devant le milieu du pied | ≤ 3 mm |
| Barre dynamique (F3, vitesse, hauteur) | **Marqueurs ArUco au moyeu des disques**, triangulés par les mêmes caméras (indépendant du détecteur) ; en option, capteur de position linéaire à câble | ≤ 5 mm |
| Angles tronc / hanche / genou (F2, F4, F5) | **Centrale inertielle sur le haut du dos**, bien sanglée : angle du tronc à ±1° y compris en mouvement (F2, F5). Centrale sur la cuisse : fiable seulement en statique (lockout, F4), à cause du mouvement de la peau. Plus une caméra sagittale plane calibrée (damier dans le plan sagittal, 120 fps, marqueurs sur repères osseux), ≈ 1–1,5° | ≤ 1,5° |
| Hauteur de hanche au placement (F8, modèle §5.4) | Marqueur sur le grand trochanter, caméra sagittale de référence | ≤ 1 cm |
| Gravité | Planche ChArUco à plat + niveau à bulle | ≤ 0,3° |
| Événements | Horodatages annotés | 1 frame |

Si la vérité terrain d'angle ne descend pas sous ≈ 1,5°, F4 et F5 restent cuées à partir de « modéré » (on ne peut pas certifier 2,7° avec un instrument à 2–3°).

### 8.5 Données synthétiques (simulateur)

Le simulateur à vérité terrain existe pour le squat dans `.claude/preik-audit/harness/preik_harness/`. Travaux :
- chemins en dur corrigés (`__init__.py:13`) ;
- `runner.py:404` rendu paramétrable ;
- générateur deadlift (≈ 200–300 lignes) : départ au sol, mains liées à la barre, `pose_from_params` réutilisé, barre au sol (aujourd'hui sur le dos dans `barbell.py`), occultation par les disques, **monde incliné**, caméra « déplacée » ;
- scénarios : propre, F1–F9, lockouts mous et en hyperextension (comptés), rep bloquée aux genoux (ratée), touch-and-go, barre lâchée, re-placement, plusieurs morphologies.

Usage : tests unitaires et d'intégration, bruit, inclinaison. **Jamais pour fixer les seuils finaux.**

### 8.6 Questions auxquelles les données doivent répondre avant la démo

1. Qualité de la pose en position penchée : keypoints perdus par caméra et par phase.
2. Occultation des pieds et tibias par la barre et les disques, par caméra.
3. Erreur réelle de chaque métrique ⇒ niveau minimal cué (§5.6) ; erreur du modèle de placement ⇒ statut de F8.
4. Bruit sur reps propres ⇒ seuils ; marge de 8 cm du lockout ⇒ comptage.
5. Précision et rappel par faute ⇒ cibles du §1.
6. Signal « raccourcissement épaule ↔ hanche » : on le garde ou on l'abandonne.

## 9. Coaching vocal, cycle de vie et messages

### 9.1 Parcours utilisateur

- **Entrée** : session « exercice rapide » (« on fait du deadlift »), ou séance programmée dont le **premier** exercice est le deadlift. Le sous-processus de pose est lancé une fois par séance avec le premier exercice (`main.py:1065-1099`). Changer d'exercice en cours de séance demande de le redémarrer : prévu en v1.1, à valider avec Ambaka car cela touche le parcours squat.
- **Refus** : sumo, RDL et autres variantes sont refusés par le code (`start_quick_exercise`), avec une réponse claire.
- **Première fois (pas de calibration `hip_hinge`)** : apprentissage deadlift (textes séparés) :
  1. pieds largeur de hanches, barre au-dessus du milieu du pied ;
  2. prise juste à l'extérieur des jambes ;
  3. tibias à la barre ;
  4. épaules au-dessus de la barre, dos plat ;
  5. mise en tension, pousser le sol.

  Quand le pipeline deadlift voit le lifter et la barre de façon fiable, il envoie `assessment_ready` avec `exercise` (transféré par `main.py`, §3.3) : la branche deadlift de l'agent d'apprentissage lance alors « vas-y, première rep », par le même callback que le squat (`coaching_service.py:609-620`). Ensuite, 2–3 reps à la barre vide (mesure du bruit, §5.5). La phase se termine par `calibration_complete` (marqueur `dl_v1`), qui active la séance côté voix comme pour le squat (`coaching_service.py:963-968`).
- **Utilisateur avec une ligne `hip_hinge` sans marqueur `dl_v1`** : traité comme une première fois (§3.3).
- **Utilisateur connu** : passage par `WorkoutAgent`, qui active la séance à l'entrée (`workout_agent.py:146`), comme pour le squat.
- **Calibration caméra** : si aucun fichier de rig n'existe, la calibration actuelle demande 2 squats lents au poids du corps. On la garde (elle calibre les caméras, pas l'exercice) et l'agent deadlift l'annonce.
- **Guidage du placement en boucle fermée** (moment fort de la démo), entièrement côté pipeline deadlift :
  - pendant `POSITION` puis `PLACEMENT`, on compare F1, F8 et F9 aux cibles ;
  - hors tolérance : cue directionnel (« avance un peu » / « avance encore », « hanches un peu plus haut ») toutes les 1,5 s au plus ;
  - dans la tolérance : « parfait » ;
  - 4 cues maximum par placement.

  C'est le même principe que le moniteur d'ajustement squat (`coaching_orchestrator.py:381-520`), réimplémenté sans le toucher.
- **Politique de cues en série** (décidée par le pipeline deadlift, seule source de vérité) :
  - jamais de cue correctif pendant le tirage ;
  - 1 correction maximum par rep, envoyée au sol ;
  - priorité : placement > perte de position (F2/F3) > lockout > asymétrie > bras ;
  - aucune correction entre deux reps en touch-and-go.

  Les cues deadlift ne passent pas par `orchestrator.on_fault`, dont l'écart minimal de 8 s entre fautes (`coaching_orchestrator.py:152`) en supprimerait une sur deux au rythme d'une rep toutes les 5–8 s.
- **Audio** : cues directionnels à deux intensités. Les chiffres (« 4 cm ») sont dits par le LLM dans le récap. Clips pré-générés hors ligne, lecture locale, repli TTS existant.
- **Cues perdus** : `_dispatch_cached_cue` abandonne les événements de plus de 1 s, et un cue attend la fin de la parole du LLM. Les cues de placement passent donc `max_age_s = 3` à `play_cached_cue`. Comme le guidage est en boucle fermée, un cue perdu est renvoyé 1,5 s plus tard tant que l'écart persiste. Chaque perte est journalisée et comptée (indicateur suivi pendant J6).
- **Récap** : déclenché comme pour le squat, par le compte de reps de l'orchestrateur, avec le diagnostic deadlift (`diagnosis_complete`) et le prompt deadlift.

### 9.2 Contrat de messages

**Pipeline deadlift → `main.py` → voix.** Tous les messages deadlift portent `exercise = "conventional_deadlift"`. Les messages squat ne changent pas (absence du champ = squat). Les noms de champs sont **ceux que le code lit déjà**.

| Type | Émis quand | Champs (noms lus par le code existant) | Consommateurs |
|---|---|---|---|
| `pipeline_status` | Démarrage, préchargement | `status` | `main.py:973` (inchangé) |
| `cache_cues` | Début de séance | `cues` (clés `dl_*`) | `coaching_service._on_cache_cues` (inchangé) |
| `assessment_ready` | Lifter et barre suivis de façon fiable, en phase d'apprentissage | `exercise` | `main.py` (transfert **ajouté**, seulement si `exercise` est présent) → `coaching_service:609` → callback de l'agent d'apprentissage (branche deadlift) |
| `assessment_rep`, `assessment_result`, `calibration_rep` | Phase d'apprentissage | mêmes champs que le squat | `main.py` (transfert existant, `:1035`), `coaching_service` (branche deadlift pour les textes) |
| `calibration_complete` | Fin de l'apprentissage | `movement_pattern = "hip_hinge"` (**explicite**, jamais le défaut « squat »), `peaks`, `thresholds`, `athlete_params` (segments, bras, décalage prise), `baseline` | `coaching_service` (sauvegarde ~915-935 et passage de la séance en actif ~963-968) |
| `dl_cue` (nouveau) | Guidage de placement **et** correction après une rep | `cue`, `kind` (`setup_correction` / `setup_confirm` / `rep_correction`), `fault_type` `dl_*`, `severity`, `severity_score`, `rep_number` (**rep courante**), `message`, `value`, `target`, `source` | `main.py` (ajout à la liste de transfert), `coaching_service` → `orchestrator.play_cached_cue` ; si `rep_correction`, `recorder.record_fault(message)`. L'accusé de livraison est fait par `_dispatch_cached_cue` lui-même (`data.fault_type`), et `_ops_cue_delivered` est idempotent |
| `rep_complete` | Barre au sol, **après** l'éventuel `dl_cue` de cette rep | `rep_number`, `set_number`, `is_clean`, `faults_in_rep` (noms), `faults_detailed` (liste {`fault_type`, `severity`, `severity_score`} de **toutes** les fautes de la rep, cuées ou non), `rep_duration_ms`, `ascent_time_s` (= durée du tirage), `descent_time_s` (= durée de la descente), `depth_category = ""`, `max_depth_angle = 0`, `rep_kinematic_summary` (métriques deadlift, JSON), `bar_source` | `coaching_service` (branche) → `orchestrator.on_rep_complete` (champs de profondeur neutres, sans effet) ; enregistreur (`_build_rep_row`, inchangé) ; affichage |
| `diagnosis_complete` | Après la série | `diagnosis` (`DiagnosisResult`), `scoring` {`mean_score`, `per_dimension` (5 dimensions deadlift), `best_rep`, `worst_rep`}, `set_number` | `coaching_service`, affichage |
| `set_complete` | Fin de série | comme le squat | **affichage seulement** (`main.py:1005`) ; ce message n'est pas transféré à la voix (`main.py:1035`) et ne lui est pas destiné |
| `rest_complete`, `frame_data` | Comme le squat | `frame_data` avec `rep_phase` deadlift | inchangé |

**Voix → pipeline deadlift**, messages existants filtrés par `main.py:601-603`, tous gérés par `deadlift/process.py` :
- `rest_start` : minuteur de repos, réarmement de la porte de départ ;
- `workout_complete` : arrêt propre ;
- `assessment_mode` : bascule apprentissage / séance ;
- `request_last_rep` et `request_demo` : en v1, pas de rejeu deadlift. Le processus répond comme le squat quand il n'a rien (`pipeline_process.py:1721-1748`) : `last_rep_snapshot {request_id, error: "no_data"}` et `demo_data_ready {request_id, status: "unavailable"}`. `_request_from_pipeline` (`coaching_service.py:340-369`) est ainsi libéré tout de suite, sans attendre son délai de 5 s ;
- `demo_start`, `demo_cue`, `demo_end` : ignorés.

**Données** : `BiomechanicsRecorder(exercise="conventional_deadlift")`. Les séances, reps et `CueEvent` sont étiquetés. Les cues `rep_correction` sont liés à leur rep et évalués sur la rep suivante, ce qui prépare l'apprentissage futur sans le construire. Les cues de placement ne sont pas enregistrés comme `CueEvent` en v1 ; les métriques de placement de chaque rep sont dans `rep_kinematic_summary`.

## 10. Edge (Jetson)

- **J1** : mesure sur Jetson Orin Nano Super de RTMPose 3 vues + YOLO11n-pose 3 vues (TensorRT FP16) + chaîne complète.
- Nouveaux calculs (machine à états, géométrie, deux résolutions du modèle, diagnostic) : solutions fermées ou recherche 1D, cible < 1 ms.
- **Mode dégradé** si le budget est dépassé : pose à 30 Hz et barre à 15 Hz avec prédiction Kalman. Sinon, tout à 15 Hz : suffisant pour le placement statique, à vérifier pour le chronométrage du décollage (§1).
- Rien de nouveau dans la boucle conversationnelle ne dépend du cloud. Annotation et entraînement : hors ligne.

## 11. Jalons, critères d'acceptation, estimations

Estimations pour 1 ingénieur à temps plein, à affiner. Avec 2 personnes, J2/J4 et J3/J4 se parallélisent.

| Jalon | Contenu | Critère d'acceptation | Dépend de | Estim. |
|---|---|---|---|---|
| **J0 — Filet squat** | `requirements.lock`, golden masters pipeline et voix, prompts, alias, manifeste de gel, test « aucune écriture squat », drapeau sans effet, CI (si validée) | Tous verts sur `main` ; une modif volontaire d'un seuil squat et d'un texte de cue squat fait échouer le bon test | — | 4–6 j |
| **J1 — Fondations** | Enregistreur + rejeu ; outil de gravité + repère lifter + tests ; `KNOWLEDGE.md` revu ; modèle de placement (2 résolutions) + tests ; capture round 1 ; mesure Jetson ; poids actuels du détecteur évalués | Revue signée ; tests de signe, d'inclinaison et de cohérence verts ; ≥ 1 h de capture brute ; keypoints de barre décidés ; chiffres Jetson | J0 | 2 sem. |
| **J2 — Simulateur deadlift** | Générateur, scénarios, monde incliné | Chaque scénario produit sa vérité terrain | J1 | 1 sem. |
| **J3 — Cœur deadlift (pose + repli poignets)** | Fournisseur caméra, `DeadliftPipeline` (contrat `CameraCalibrationSession`), machine à états, références debout, métriques F1–F9, guidage de placement, golden master deadlift | Simulateur : 100 % des reps comptées (lockouts mous et hyperextension compris), reps bloquées aux genoux non comptées, événements ≤ 100 ms, chaque faute injectée détectée, scénario propre sans faute ; rejeu round 1 sans crash | J2 | 2 sem. |
| **J4 — Barre 3D** | Annotation CVAT, entraînement, export, tracker 3D, association multi-vues, vérité terrain ArUco | Rappel ≥ 95 % ; erreur 3D statique ≤ 1 cm, dynamique ≤ 1,5 cm vs ArUco ; temps Jetson dans le budget ou mode dégradé validé | J1 | 2–3 sem. |
| **J5 — Intégration** | `deadlift/process.py`, bifurcation, contrat §9.2 dans les deux sens, fichiers §3.3, audio `dl_*`, affichage, base | Séance complète sur le rack, première fois **et** utilisateur connu : « deadlift » → placement guidé → reps comptées → cue → récap, données en base avec liaison cue ↔ rep ; ligne `hip_hinge` de forme squat ⇒ parcours première fois ; tests d'isolation des données verts ; golden masters squat identiques drapeau éteint et allumé | J0, J3, J4 | 2 sem. |
| **J6 — Validation réelle** | Annotation round 1 → seuils, budget d'erreur, marge de lockout ; round 2 → jeu de test | Cibles du §1 sur ≥ 8 lifters jamais vus (bootstrap par lifter, plancher par lifter, séries naturelles à part) ; κ ≥ 0,6 ; statut de chaque faute (héros / provisoire / désactivée) | J5 | 3–4 sem. |
| **J7 — Diagnostic deadlift** | Graphe statique, score, récap | Cause n°1 = annotation du coach dans ≥ 70 % des séries annotées | J6 | 1–1,5 sem. |
| **J8 — Démo** | Scénario héros : placement guidé (F1, puis F8 si elle est certifiée) puis F2 ou F3 corrigée à la rep suivante. Plan B : F1 + F3 seulement | 10 démos consécutives sans erreur ; budget Jetson respecté | J7 | 1 sem. |

Total indicatif : 16–18 semaines pour 1 personne, ≈ 10 semaines pour 2. La capture réelle démarre dès J1, en parallèle du code.

## 12. Risques et parades

| Risque | Parade |
|---|---|
| Toucher au squat par accident | Sous-processus séparé, composition, drapeau, golden masters pipeline + voix, manifeste de gel, test « aucune écriture squat », CI |
| Outil deadlift qui écrit un état squat | Outil de gravité dédié, `HOME` temporaire dans les tests |
| Axe vertical faux / caméra déplacée | Gravité par caméra + contrôle de cohérence, tests d'inclinaison, repli élargi |
| Dos rond non mesurable | Proxys, vocabulaire comportemental, signal expérimental soumis aux données |
| Pose dégradée en position penchée | Mesure au round 1 ; affinage RTMPose si besoin (hors v1) |
| Occultation des pieds et tibias | Milieu du pied verrouillé avant le tirage ; mesure par caméra |
| Détecteur de barre qui ne généralise pas | Données variées, découpage par lifter, négatifs |
| Seuils inventés | Valeurs initiales seulement, budget d'erreur, données réelles avant la démo |
| Jeu de test trop petit | Dimensionnement par Wilson, fautes scénarisées, statut provisoire |
| Annotations incohérentes | Double annotation, κ, arbitrage |
| Modèle de placement imprécis | Critère P95 explicite, F8 dégradée ou désactivée, plan B de démo |
| Budget Jetson | Mesure à J1, mode dégradé |
| Duplication qui diverge | Golden master deadlift, tests de contrat d'interface |
| Prise mixte qui fausse l'asymétrie | Rotation retirée de F6, prise enregistrée |
| Collisions de noms en base | Préfixe `dl_`, pas d'entrée dans `END_OF_REP_FAULT_TYPES` |
| Ancienne ligne `hip_hinge` de forme squat | Marqueur de provenance `dl_v1`, test |
| Validation trop optimiste | Bootstrap par lifter, plancher par lifter, séries naturelles séparées |

## 13. Hors périmètre v1

Sumo, RDL et autres variantes ; changement d'exercice en cours de séance ; apprentissage des probabilités ou du LLM ; fantôme animé ; ML (TCN) pour le dos rond ; VBT complet ; rejeu « dernière rep » deadlift ; toute modification d'un fichier squat.

## 14. Questions pour Ambaka

1. Qui annote (Ambaka, coach externe) ?
2. Touch-and-go compté en v1 (proposé : oui, sans correction entre les reps) ?
3. Où sont les poids actuels du détecteur de barre, et sur quelles images ont-ils été entraînés ?
4. OK pour la CI GitHub Actions ?
5. OK pour la duplication assumée (boucle de session, moteur de diagnostic) ?
6. Accéléromètre dans le boîtier envisageable ?
7. Quels disques dans la salle de démo ?
8. Observations hors périmètre, à vérifier côté squat (on n'y touche pas) :
   - l'axe vertical du monde n'est pas la gravité (§2.1) ;
   - `main.py:1035` ne transfère pas `assessment_ready` ni `shallow_rep` à la voix, alors qu'elle les gère (`coaching_service.py:524, 609`).
