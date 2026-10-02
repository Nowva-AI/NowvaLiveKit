# Plan — Soulevé de terre conventionnel (deadlift) — v1

Statut : proposition, à valider par Ambaka. Version 9 du document (notée 9,0/10 par un relecteur indépendant).

## 0. Décisions déjà prises (ne pas rediscuter)

| Décision | Conséquence pour le plan |
|---|---|
| On sort du « squat only » de `CLAUDE.md` pour le deadlift | Chantier autorisé, mais il ne doit rien coûter au squat |
| **Le squat ne doit jamais être modifié et reste séparé du deadlift** | Propre sous-processus et propre package pour le deadlift. Aucun fichier squat modifié, aucun fichier d'état du squat écrit — **calibration caméra comprise** (§3.1). Les fichiers partagés (voix, affichage) ne reçoivent que des ajouts derrière un drapeau. Deux filets de tests (pipeline et voix) prouvent que le squat ne bouge pas |
| Deadlift **conventionnel uniquement** | Sumo et variantes refusés **dans le code** |
| Dos rond : proxys indirects acceptés | On ne prétend jamais mesurer la colonne |
| Les caméras du rack voient la barre au sol | Détection et suivi 3D de la barre au sol à construire |
| Apprentissage du LLM : hors périmètre | On enregistre seulement les données, étiquetées par exercice |
| **Piège des axes** | §2 — règle bloquante, y compris pour les sessions de code assistées par IA |

## 1. Objectif produit et critères de succès

Boucle démo YC deadlift :

> « On fait du deadlift » → Nova guide la position des pieds **en boucle fermée** (« avance un peu… encore… parfait ») → l'athlète se met en place et tire → le système détecte une faute mesurable → **un** cue au bon moment (au sol, avant la rep suivante) → rep suivante corrigée → récap de série chiffré.

Fautes « héros » (démo) : F1, F2, F3, F8. Les autres fautes v1 sont « provisoires » : cuées seulement à partir de « modéré », jusqu'à ce que les données suffisent (§8.3).

| Métrique (données réelles, lifters jamais vus) | Porte démo (J6) | Porte lancement client (après ce plan) |
|---|---|---|
| Comptage des reps | ≥ 99 %, 0 rep fantôme sur séries propres | idem |
| Événements (décollage, passage des genoux, haut, sol) | erreur médiane ≤ 100 ms | idem |
| Précision par faute héros | ≥ 0,85 ; borne basse à 95 % (bootstrap par lifter) ≥ 0,70 ; ≥ 0,80 sur les séries naturelles | borne basse ≥ 0,75 |
| Rappel par faute héros | ≥ 0,70 ; borne basse ≥ 0,55 | borne basse ≥ 0,60 |
| Fausses corrections sur reps propres | ≤ 1 pour 10 reps | idem |
| Erreur de mesure | ≤ 1/3 du seuil du niveau cué (§5.6) | idem |
| Latence d'un cue | clips pré-générés, joués en local ; aucune synthèse de clip pendant la séance (`generate_tts` / `_generate_single_cue`) | idem |
| Squat | golden masters pipeline **et** voix identiques ; aucun fichier d'état squat écrit | idem |
| Edge | ≤ 33 ms/frame sur Jetson Orin Nano Super, ou mode dégradé défini (§10) | idem |

## 2. Repères et piège des axes (règle bloquante)

### 2.1 Ce que dit le code

- Repère monde en mètres, **Y vers le bas** (un Y plus grand est plus bas), X = gauche du sujet, **+Z = dos du sujet** (l'avant est donc −Z). Sources : `triangulation/person_calibration.py:5`, `triangulator.py:2`, `utils/foot_contact.py:32`, `kinematics/analytical_ik.py:449`.
- Deux documents disent le contraire du code :
  - `.claude/rules/biomechanics.md` écrit « Larger Y = higher ». Ce fichier est chargé automatiquement pour tout `src/biomechanics/**`, donc aussi pour le futur package deadlift.
  - `README.md:325` écrit « Z-forward ».
- **L'axe Y du monde n'est pas la gravité.** `world_frame_from_standing` (`person_calibration.py:734-752`) prend pour Y la direction médiane hanche → cheville en position debout, ce qui peut s'écarter de 2–4° de la verticale. Sur 55 cm de course de barre, 3° d'écart créent ≈ 2,9 cm de fausse dérive avant/arrière, soit le seuil de F3.

**Livrable J0** : une règle `.claude/rules/deadlift.md`, limitée à `src/biomechanics/deadlift/**` et `tests/test_deadlift/**`. `biomechanics.md` (glob `src/biomechanics/**`) s'applique aussi au package : `deadlift.md` dit donc explicitement qu'elle **remplace** sa ligne sur l'axe Y et ses conventions `faults/rules/`. Elle rétablit les bons axes et fixe les conventions du package : les règles de fautes vont dans `deadlift/rules/` et non dans `faults/rules/`, et le repère passe par `frame.py`. En parallèle, on propose à Ambaka de corriger la ligne fausse de `biomechanics.md` (documentation seulement, sans effet sur le code squat).

### 2.2 Repère du lifter

Aucune métrique deadlift n'utilise Y ou Z bruts : tout passe par `src/biomechanics/deadlift/frame.py`. Les sorties de l'IK ne servent qu'aux **angles articulaires relatifs** (coude, genou, hanche entre segments). Tout angle par rapport à la verticale est calculé dans `frame.py`.

**Vertical = gravité.** Trois sources, dans cet ordre de préférence :

- **(a) Mesure de gravité propre au deadlift** — outil `scripts/tools/deadlift_gravity.py` (nouveau), planche ChArUco posée à plat au sol :
  - pour **chaque caméra**, on résout la pose de la planche (`solve_board_pose`, `charuco.py:334`, par import) et on en tire sa normale dans le repère de la caméra ;
  - contrôle de cohérence : on ramène ces normales dans le monde avec les rotations de la calibration deadlift (§3.1). Si elles s'écartent de plus de 0,5°, l'outil refuse ;
  - seule écriture : `~/.nowva/deadlift/gravity_<ids>.json`. L'outil n'appelle jamais `calibrate_cameras.py extrinsics`, qui sauvegarde `rig_calibration_*.json` (`calibrate_cameras.py:338-343`) ;
  - une seule mesure en usine sur le rack produit ; à refaire après chaque déplacement des trépieds sur le rig de dev.
- **(b) À l'exécution** : la gravité monde est la moyenne robuste, sur les caméras, de `R_caméra→monde · g_caméra`, calculée avec la calibration courante. Une caméra qui s'écarte de plus de 1° est exclue et signalée. S'il reste moins de 2 caméras cohérentes, on passe au repli.
- **(c) Repli** : axe hanche → cheville, avec `gravity_source = "body"`. F2, F3 et F5 ne sont alors cuées qu'à partir de « modéré ».

Option matérielle à discuter : un accéléromètre (~1 $) dans le boîtier.

**Autres axes et conversions :**
- Latéral = axe de la barre au repos projeté à l'horizontale (ou ligne des hanches sans barre). Avant = produit vectoriel, orienté talon → pointe.
- Conversions uniques : `height_m`, `forward_of_m`, `lateral_of_m`, `sagittal_angle_deg` (signé, positif vers l'avant).
- **Les métriques sont des différences** : barre vs son repos, poignets vs leur hauteur debout, barre vs milieu du pied. Rien ne dépend du sol absolu (§6.5).

**Tests obligatoires :**
- signe : la barre monte pendant le tirage ; une barre devant le pied donne « avant » positif ;
- inclinaison : un monde synthétique incliné de 3° et 5° avec la gravité fournie reste exact à ±2 mm et ±0,5° ;
- repli : l'erreur est rapportée et les seuils sont élargis ;
- cohérence : une caméra « déplacée » de 2° est exclue ;
- **test statique** (sur l'arbre syntaxique) : aucune soustraction de `.y`/`.z` ni indexation `[..., 1]`/`[..., 2]` sur les **tableaux 3D du monde** dans `src/biomechanics/deadlift/`, sauf dans `frame.py`. Les tableaux 3D du monde portent un suffixe de nom imposé (`*_world`) : c'est ce qui rend le test précis. Un second contrôle de l'arbre syntaxique impose cette convention : tout résultat 3D de la triangulation ou du tracker de barre, ainsi que l'élément « squelette 3D » du tuple renvoyé par `get_frames_and_pose()`, doit être affecté à un nom en `*_world`. Les fichiers qui manipulent des pixels ou des confiances (`bar_detector.py`, DLT) figurent sur une liste d'exceptions explicite.

## 3. Isolation du squat (contrainte dure)

### 3.1 Architecture

```
main.py (~l. 452) : deadlift conventionnel (reconnu par deadlift/voice/names.py) ET NOWVA_ENABLE_DEADLIFT ?
   ├── non → pipeline_process.py (squat, inchangé, 0 ligne modifiée)
   └── oui → src/biomechanics/deadlift/process.py (nouveau sous-processus, mêmes arguments, §9.3)
```

Le package est construit **par composition**. Il n'hérite pas de `BiomechanicsPipeline`, qui chargerait le profil placeholder, le moteur de règles et le compteur squat.

| Module | Rôle | Réutilise (import, lecture seule) |
|---|---|---|
| `process.py` | Boucle de session, IPC dans les deux sens, démarrage et arrêt (§9.3) | `CameraCalibrationSession` (avec des chemins deadlift) ; helpers de `pipeline_process.py` importés s'ils sont au niveau du module, sinon recopiés avec un commentaire d'origine |
| `pipeline.py` (`DeadliftPipeline`) | Par frame : pré-IK → IK → repère lifter → barre 3D → machine à états → métriques → règles | `build_preik_chain`, `AnalyticalIK`, `SegmentLengthEstimator` |
| `camera_provider.py` | Sous-classe de `MultiCameraPoseProvider` qui **ajoute** `get_frames_and_pose()` | `MultiCameraPoseProvider`, `RTMPoseEstimator`, `DLTTriangulator` |
| `frame.py` | Repère lifter (§2.2) | fichiers de calibration deadlift et de gravité |
| `bar_detector.py`, `bar_tracker_3d.py` | Barre multi-vues et 3D (§6) | `calibration.undistort_keypoints`, détecteur (§6.3, licence) |
| `rep_counter.py`, `metrics.py`, `rules/` | Phases, métriques, fautes | — |
| `setup_model.py` | Placement attendu (§5.4) | — |
| `ipc.py` | Messages (§9.2), politique de cues | client IPC existant |
| `diagnosis/` | Graphe statique (§7) | type `DiagnosisResult` |
| `voice/` | Textes deadlift, `names.py`, `calibration.py` (provenance), textes et chemins audio des cues | — |
| `config/deadlift.yaml` | Seuils et paramètres | `config/biomechanics.yaml` intact |

**Calibration caméra du deadlift : une copie séparée, jamais d'écriture dans les fichiers du squat.**

`CameraCalibrationSession` écrit les fichiers du rig à quatre endroits :
- calibration initiale à partir du lifter (`pipeline_process.py:681`, via `_save_and_install`) ;
- T-pose (`:712`) ;
- la sauvegarde générique `_save_and_install` (`:741`), utilisée aussi bien par l'amorçage que par l'affinage ;
- l'appel unique de fin d'affinage (`:787`, vers `_refined`), commun à l'affinage de dérive et au ré-ancrage du monde.

Un affinage pendant une séance deadlift (positions penchées, pieds masqués par les disques) produirait un `_refined` que `select_calibration_file` chargerait ensuite **pour le squat**. Comme le constructeur reçoit `factory_path` en paramètre (`pipeline_process.py:495-505`), le deadlift lui passe ses propres chemins :
- `factory_path = ~/.nowva/deadlift/rig_calibration_cams_<ids>.json`, avec son `_refined` à côté ;
- **amorçage par paire** : si le fichier deadlift n'existe pas, on **copie** en lecture seule le fichier d'usine du squat **et** son `_refined`, en conservant leurs champs `timestamp` / `source_timestamp`. `select_calibration_file` (`pipeline_process.py:367-381`) choisit le `_refined` seulement si son `source_timestamp` égale le `timestamp` du fichier d'usine. Copier un `_refined` seul comme fichier d'usine casserait ce lien : les affinages deadlift, sauvés avec `source_timestamp` = l'horodatage d'usine (`:549`, `:733-734`), seraient ignorés à chaque séance. En copiant la paire, `establish()` reprend la même chaîne et les affinages deadlift suivants sont bien rechargés. Un test vérifie que l'affinage deadlift d'une séance est celui chargé à la séance suivante ;
- **ré-amorçage** : `~/.nowva/deadlift/seed.json` mémorise l'horodatage du fichier d'usine squat copié. On ré-amorce **uniquement** quand cet horodatage change (caméras recalibrées côté squat). Un échec du contrôle de cohérence de la gravité signifie qu'une caméra a bougé physiquement. Re-copier la calibration squat, peut-être tout aussi périmée, n'y changerait rien. La caméra est donc exclue (§2.2 b), et le pipeline envoie `dl_status {code: "camera_moved"}` (§9.2) : Nova demande alors une recalibration puis une nouvelle mesure de gravité. S'il n'y a aucun fichier squat, la calibration à partir du lifter écrit dans le dossier deadlift, et le ré-amorçage ne s'applique pas ;
- affinages, ré-ancrages et sauvegardes se font uniquement dans `~/.nowva/deadlift/`.

Conséquence acceptée : les calibrations squat et deadlift peuvent diverger avec le temps, chacune avec son propre contrôle de dérive. Partager un jour la calibration du rig sera une décision d'Ambaka.

**Contrats d'interface** (testés à J3) :
- `DeadliftPipeline` expose ce qu'utilise `CameraCalibrationSession` : `_multi_camera_provider`, `presence_only`, `process_frame`, `last_frame`, `on_calibration_changed`, `body_calibration`.
- `get_frames_and_pose()` utilise `_capture.get_synced_frames()`, `_estimator.estimate_batch`, `_triangulator`, `_min_views` et `_primary_camera`. Elle appelle `_record_views()` comme `get_pose()` (`multi_camera.py:307-357`). Un test compare ses sorties à celles de `get_pose()`.

Autres points :
- Le BiLSTM, entraîné sur des squats (`pipeline.py:243-259, 815-854`), n'est jamais instancié.
- Le placeholder `profiles/deadlift.py` n'est jamais chargé quand le drapeau est actif.
- La duplication est acceptée (boucle de session, ≈ 150 lignes de calcul bayésien). Une factorisation ne se fera qu'avec l'accord d'Ambaka et sous golden master.

### 3.2 Drapeau

`NOWVA_ENABLE_DEADLIFT` (défaut `false`). Drapeau éteint, les fichiers partagés se comportent exactement comme aujourd'hui, prompts et cas d'erreur compris. Par exemple, « deadlift » lance aujourd'hui le pipeline squat avec le profil placeholder : c'est conservé.

**Retour arrière** (drapeau éteint après usage du deadlift) : avec une ligne `hip_hinge` marquée `dl_v1`, le chemin placeholder ne doit pas se dégrader.
- `apply_calibration_to_rule_engine` exige les clés des règles du placeholder (ex. `profile["bilateral_asymmetry"]` pour sa `SymmetryRule`, `calibration.py:124-128`).
- Une erreur `KeyError` serait attrapée à `pipeline_process.py:1066`, mais elle sauterait aussi le chargement des mesures corporelles du même bloc (`:1050-1056`).
- Les lignes `dl_v1` contiennent donc, en plus des clés `dl_*`, les clés attendues par le placeholder, **avec les valeurs par défaut qu'il utiliserait** (`bilateral_asymmetry` etc.).
- `athlete_params` deadlift contient au moins toutes les clés du squat.
- Test : drapeau éteint + ligne `dl_v1` ⇒ pas d'exception, seuils et mesures corporelles chargés.

### 3.3 Fichiers partagés modifiés : liste complète

Pour établir cette liste, on a suivi chaque message (dans les deux sens), chaque recherche de calibration, chaque lecture et écriture en base, l'audio et l'écran. Tous les changements sont des **ajouts** : soit derrière le drapeau, soit déclenchés uniquement par des messages que le squat n'émet jamais (champ `exercise`, types `dl_*`).

| Fichier | Changement |
|---|---|
| `src/main.py` ~452 | Choix du script de sous-processus. Le nom, parfois brut (« Barbell Conventional Deadlift » en séance programmée, `main_menu_agent.py:85`), est reconnu par `names.py` |
| `src/main.py` ~1060-1125 | Branche deadlift du lancement : charge les mesures corporelles **dans tous les cas**, première fois comprise. Aujourd'hui `_load_athlete_calibration` n'est appelée que pour un utilisateur connu (`:1074-1081`). Source : ligne `hip_hinge` `dl_v1`, sinon `athlete_params` de la ligne `squat` en lecture seule. Joint les métadonnées de séance (§9.3) au message `start_capture` (`:1104-1118`), envoyé après `workout.greeting_done` |
| `src/main.py` ~989-1035 | Branche d'affichage pour les messages qui portent `exercise`. Transfert vers la voix de `dl_cue`, `dl_diagnosis_update`, `dl_status`, et de `assessment_ready` **seulement s'il porte `exercise`** |
| `src/visual/display.html` ~963, 1223, 1317, 1422 | Tuiles et libellés deadlift si `exercise` est présent |
| `src/agent/agents/prompts/main_menu_prompt.py:26` | Texte « squats et deadlift conventionnel » si drapeau |
| `src/agent/agents/main_menu_agent.py` | Si drapeau : `start_quick_exercise` refuse les variantes non prises en charge ; branche deadlift (`names.py`) qui passe le nom canonique `"Barbell Deadlift"` à `check_calibration` / `start_calibration_mode` et vérifie la provenance ; `start_workout` (~84-119) passe d'abord par l'apprentissage deadlift si c'est la première fois ; texte de progression (~640) adapté à l'exercice |
| `src/agent/agents/quickExerciseAgent.py` | `:211` passe `exercise=` ; branche deadlift à la décision de calibration (`:177-190`, provenance) ; branche deadlift de `CollectExerciseInfoTask` (`:76`) qui demande aussi le type de prise et de disques (§9.3) |
| `src/agent/agents/teaching_agent.py`, `calibration_agent.py`, `workout_agent.py` | Retour anticipé vers `deadlift/voice/` si l'exercice est le deadlift ; code squat ni modifié ni réindenté |
| `src/agent/services/coaching_service.py` | **Enregistrement** : dans `_start_biomech_recording` (~113-147), si drapeau et `workout.exercise_name` reconnu comme deadlift, `BiomechanicsRecorder(exercise="conventional_deadlift")`. Le squat ne passe rien (défaut `"squat"`), donc rien ne change ; le paramètre existe déjà (`biomechanics_persistence.py:290`). **Messages** : branches `rep_complete` / `diagnosis_complete` / `calibration_complete` / `assessment_ready` quand `exercise` est présent. Nouveaux gestionnaires `dl_cue` (plafond compté sur les cues réellement joués, §9.1 ; publie lui-même le bandeau d'écran avec le texte deadlift, car `_publish_cue_banner` lit `CUE_TEXT_MAP`, `:1178`) et `dl_diagnosis_update` (appelle `set_diagnosis_data` **avant tout `await`**, chaque message IPC tournant dans sa propre tâche). **Progression** : requêtes avec `exercise="conventional_deadlift"` (~159-160). **Prompts** deadlift |
| `src/agent/services/coaching_orchestrator.py` | Nouvelle méthode publique `play_cached_cue(cue_key, priority, data)`, qui place un événement `cached_cue` dans la file existante ; jamais appelée par le squat. `_dispatch_cached_cue` **n'est pas modifié**, y compris l'abandon des cues de plus d'1 s (`:966`). Branches de récap deadlift (~1227-1251, 1388-1390, 1407, ligne d'ajustement ~1424-1428, 1510). Le chiffre de la correction est formaté par `deadlift/diagnosis/parameter_deltas.py`, car `summarize_cue_magnitude` renvoie un texte générique pour les causes `dl_*` (`demo_builder.py:77-79`) |
| `src/agent/services/audio_cue_service.py` | Nouvelle méthode `load_extra_cues(directory, texts)`, appelée **seulement en séance deadlift**. Elle charge les clips de `src/assets/cues/wav_deadlift/` et leurs textes. Les clips deadlift ne sont donc pas préchargés pendant les séances squat (`_load_from_disk`), et `CUE_TEXT_MAP` / `_validate_expected_cues` ne changent pas |
| `src/agent/services/progress_context.py` | Libellés et filtre par exercice |
| `scripts/tools/generate_cue_audio.py` | Prompts audio deadlift, sortie dans `wav_deadlift/` |

**Jamais modifiés** :
- pipeline : `pipeline.py`, `pipeline_process.py`, `profiles/*`, `diagnosis/*`, `faults/*`, `coaching/{cue_cache,ipc_bridge,session_tracker}.py`, `calibration.py`, `utils/foot_contact.py`, `pose/*`, `barbell_tracking/*` ;
- configuration et outils : `config/biomechanics.yaml`, `db/biomechanics_persistence.py`, `scripts/tools/calibrate_cameras.py` ;
- voix : `agent/agents/shared/helpers.py`, `agent/services/coaching_constants.py`.

**Motif de mouvement** : on garde la clé existante `hip_hinge`, déjà associée à `"Barbell Deadlift"` (`calibration.py:28`). Les recherches de calibration existantes fonctionnent donc sans modifier `calibration.py`. Le RDL et le sumo partagent cette clé, mais ils sont refusés en v1 ; il faudra une clé distincte et une migration s'ils arrivent un jour.

**Provenance de la calibration `hip_hinge`** : drapeau éteint, « deadlift » ou « RDL » en parcours rapide lancent le pipeline squat avec le placeholder. Celui-ci écrit sous `hip_hinge` un profil **de forme squat** (`pipeline_process.py:1073, 1499`). D'où les règles suivantes :
- chaque calibration écrite par le deadlift porte `thresholds.schema = "dl_v1"` ;
- une ligne sans ce marqueur est traitée comme **« pas de calibration deadlift »** : l'apprentissage a lieu, et seules les mesures corporelles servent de point de départ ;
- `deadlift/voice/calibration.py::is_deadlift_calibration(row)` est appelée par les branches deadlift de `main_menu_agent`, `quickExerciseAgent` et `main.py` ;
- tests : une ligne de forme squat ⇒ première fois ; une ligne `dl_v1` ⇒ utilisateur connu.

**Isolation des données** :
- les lectures du squat filtrent déjà sur `exercise="squat"` (`get_last_completed_session`, `get_progress_baseline`, `get_score_progress`, `get_multi_session_fault_trends` ; `db/biomechanics_persistence.py:738-830, 922+`) ;
- les fonctions non filtrées (`get_fault_progress`, `get_rep_kinematics_history`, `get_cue_effectiveness`) n'ont aucun appelant ; tout futur appelant devra filtrer par exercice ;
- `rep_kinematic_summary` deadlift porte `dl_schema: 1` et `exercise`.

**Préfixe `dl_`** sur tous les types de faute et toutes les clés de cue. Les types `dl_*` ne sont **pas** ajoutés à `END_OF_REP_FAULT_TYPES` : ils sont émis avec la rep courante avant `rep_complete`, donc la liaison et l'évaluation cue ↔ rep fonctionnent (`biomechanics_persistence.py:467-489`).

### 3.4 Filet de sécurité squat (J0, avant toute ligne deadlift)

1. **Golden master pipeline squat** (`tests/test_biomechanics/test_squat_golden.py`) :
   - scénarios rejoués via le harnais de `tests/test_biomechanics/test_pipeline.py:329-392` (`_FakeProvider`, `_FakeClock`) : propre, valgus, inclinaison, `go_deeper`, perte de détection, asymétrie ;
   - mono-caméra (`estimate_both` simulé) et BiLSTM (faux déterministe) ;
   - on capture les sorties par frame (`model_dump` sans `latency_ms`), puis `SessionTracker` + `IPCBridge` simulé + `force_end_set`, avec tous les messages IPC ;
   - `time.time` remplacé dans `session_tracker` et `ipc_bridge`, `random` initialisé ;
   - sérialisation canonique (arrondi à 1e-9), environnement `requirements.lock`, drapeau éteint et allumé.
2. **Golden master voix squat** (`tests/test_coaching_squat_golden.py`, au même niveau que `tests/test_coaching_service.py`, qui construit déjà `CoachingService(session=None, state=None)`) :
   - le flux du point 1 est rejoué dans `CoachingService` + `CoachingOrchestrator`, avec TTS, LLM et base simulés ;
   - on capture les cues joués, le texte des prompts de récap et les opérations de l'enregistreur (`_ops_*`) ;
   - `time.monotonic` remplacé, `random` initialisé (`audio_cue_service.py:344`), file vidée entre deux messages ;
   - drapeau éteint et allumé.
3. **Golden master des prompts squat.**
4. **Alias** : alias squat → `SquatProfile` et motif `squat` ; avec le drapeau, sumo refusé.
5. **Manifeste de gel** (`tests/test_squat_freeze.py`) : SHA-256 de la liste « Jamais modifiés ».
6. **Aucune écriture d'état squat** :
   - une séance deadlift complète est rejouée dans un `HOME` temporaire, **avec un affinage de dérive et un ré-ancrage forcés**. L'empreinte et la date de modification de tous les fichiers de `~/.nowva/` hors `deadlift/` doivent rester identiques, ainsi que celles de `config.triangulation.calibration_file` quand il est défini ;
   - variante base de données : avec des séances squat et deadlift en base, le contexte squat (accueil, progression, récap) est identique à celui d'une base squat seule ;
   - variante drapeau éteint avec une ligne `dl_v1` : pas de plantage, chemin placeholder inchangé.
7. **Rejeu de vraies séances squat** : dès que l'enregistreur existe (J1), 3–5 séances squat brutes sont rejouées dans le pipeline squat inchangé, via un fournisseur de rejeu injecté dans le test. Aujourd'hui, `user_test_runs/` ne contient que des sorties.
8. **CI** (il n'y en a aucune aujourd'hui) : GitHub Actions avec `requirements.lock`, `PYTHONPATH=src pytest tests/ -x` sur chaque PR, à valider.
9. **Golden master deadlift** (J3) : protège le deadlift contre de futurs changements du code partagé.

Preuve du filet : modifier volontairement un seuil squat ou un texte de cue squat doit faire échouer le bon test.

## 4. Le mouvement : phases et machine à états

### 4.1 Phases

1. `APPROCHE` : debout.
2. `POSITION` : debout au-dessus de la barre, pieds plantés.
3. `PLACEMENT` : penché, mains sur la barre, immobile.
4. `TIRAGE` : décollage → passage des genoux → fin du tirage.
5. `HAUT`.
6. `DESCENTE`.
7. `AU SOL`.

Ensuite, la rep suivante repart par `PLACEMENT`, ou directement par `TIRAGE` ; sinon on revient en `APPROCHE`.

### 4.2 Références personnelles prises à l'approche

Pendant `APPROCHE`, debout immobile (≥ 1 s), on enregistre :
- les angles sagittaux **signés** de hanche, genou, tronc et coude ;
- la hauteur des poignets bras pendants. C'est la **hauteur de haut attendue** : telle quelle en mode poignets, et moins le décalage poignet → barre en mode barre ;
- les ancres et le milieu du pied (§6.5).

F4, F5 et F7 sont mesurées par rapport à ces références.

### 4.3 Signal de rep

Hauteur du centre de la barre, relative à sa hauteur de repos. En repli, milieu des poignets, relatif à sa hauteur en placement. On n'inverse jamais le signal de hanche comme le faisait le placeholder.

### 4.4 Machine à états (`deadlift/rep_counter.py`)

Le comptage est **piloté par la barre** ; la posture est jugée par les fautes.

| Transition | Condition (valeurs initiales, `config/deadlift.yaml`) |
|---|---|
| `APPROCHE → POSITION` | Debout, milieu du pied à moins de 15 cm horizontaux de la barre, pieds immobiles ≥ 0,5 s |
| `POSITION → PLACEMENT` | Mains à moins de 10 cm au-dessus de la barre, immobiles ≥ 0,3 s |
| `PLACEMENT → POSITION` / `APPROCHE` | Le lifter se relève sans soulever |
| `PLACEMENT → TIRAGE` | Déclencheur : barre > repos + 3 cm et vitesse verticale > 0,10 m/s |
| `TIRAGE → HAUT` | Barre ≥ hauteur de haut attendue − 8 cm, \|v\| < 0,05 m/s pendant ≥ 3 frames, tronc à moins de 35° de la verticale |
| `TIRAGE → AU SOL` sans passer par `HAUT` | **Rep ratée** (la barre redescend sous la hauteur de haut − 8 cm) : événement, non comptée |
| `HAUT → DESCENTE` | Vitesse < −0,10 m/s |
| `DESCENTE → AU SOL` | **Arrêt au sol** : barre ≤ repos + 2 cm et \|v\| < 0,02 m/s pendant ≥ 3 frames, **ou** barre restée dans repos + 2 cm pendant ≥ 0,3 s (repli robuste au bruit de la vitesse filtrée) → **rep comptée et analysée** |
| `DESCENTE → TIRAGE` | **Touch-and-go** : point bas de la barre à moins de 5 cm de son repos, puis remontée de plus de 3 cm au-dessus de ce point bas, vitesse positive soutenue ≥ 100 ms, mains sur la barre → **la rep qui s'achève est comptée et analysée à cet instant** (`dl_diagnosis_update`, puis `rep_complete`). La rep suivante commence, avec pour décollage le point bas |
| **`AU SOL → PLACEMENT`** | **Arrêt complet** : mains toujours sur la barre, immobiles ≥ 0,3 s. Nouveau placement analysé : F1, F8 et F9 sont remesurées |
| **`AU SOL → TIRAGE`** | **Relance rapide sans pause** : même déclencheur que `PLACEMENT → TIRAGE`. Le placement est mesuré sur les 0,2 s qui précèdent le décollage, ou marqué « non mesuré » si c'est trop court |
| `AU SOL → POSITION` / `APPROCHE` | Le lifter se relève, lâche la barre ou recule (fin de série possible) |

Précisions :
- **Décollage daté au début réel du mouvement.** Sur un tirage lent, le déclencheur (3 cm et 0,10 m/s) arrive ≈ 150 ms après le vrai début. On remonte donc dans un tampon de 0,5 s jusqu'à la dernière frame où la barre était à moins de 0,5 cm de son repos avec une vitesse < 0,02 m/s. C'est cet instant qui sert de décollage pour l'événement, pour la fenêtre de F2 et pour la durée du tirage.
- La marge de 8 cm sert à compter les lockouts mous ou en hyperextension, que F4/F5 signalent ensuite. Une rep bloquée aux genoux n'est pas comptée. La marge est validée sur données réelles (§8).
- Une rep est comptée **une seule fois**, par l'une de deux transitions : `DESCENTE → AU SOL` (arrêt au sol) ou `DESCENTE → TIRAGE` (touch-and-go). Chacune émet un seul `rep_complete` (§9.2). Un rebond de bumper (≤ 3 cm) ne déclenche pas de touch-and-go : la barre finit par s'arrêter et la rep est comptée en `AU SOL`. Si le lifter enchaîne un vrai tirage juste après un rebond, la rep est comptée une seule fois, par `DESCENTE → TIRAGE`, car l'arrêt au sol n'a jamais été validé.
- Barre lâchée depuis le haut : rep comptée, pas de faute. Ajustements de la barre au sol de moins de 3 cm : ignorés.
- **Hystérésis au sol** : l'arrêt au sol se valide à repos + 2 cm avec \|v\| < 0,02 m/s pendant ≥ 3 frames ; le décollage se déclenche à + 3 cm. Un rebond de bumper (1–3 cm, sans mains qui tirent) n'est pas un décollage : il faut une vitesse montante soutenue ≥ 100 ms, mains sur la barre. Une « rep ratée » n'est enregistrée que si la barre est montée d'au moins 10 cm.
- Fin de série : `set_timeout_seconds` (30 s) en `AU SOL` ou `APPROCHE`, « j'ai fini », ou décision de la voix (§9.3).
- La hauteur de repos est la médiane pendant `POSITION`/`PLACEMENT`, recalculée à chaque série.
- En touch-and-go, aucune correction entre deux reps.

## 5. Fautes, mesures, seuils

Toutes les mesures se font dans le repère lifter (§2.2). Les seuils initiaux sont dans `config/deadlift.yaml` et seront fixés sur données réelles (§8). Il y a trois niveaux : léger, modéré, sévère. Le niveau minimal cué dépend du budget d'erreur (§5.6) et du statut de la faute.

### 5.1 Fautes v1

| # | Faute | Phase | Mesure | Seuils initiaux | Cue (moment) |
|---|---|---|---|---|---|
| F1 ★ | **Barre pas au-dessus du milieu du pied** | `POSITION` (debout), recontrôle en `PLACEMENT` | `bar_forward_of_midfoot_cm` | 3 / 5 / 8 cm | **Boucle fermée** en `POSITION` : « avance un peu / encore » → « parfait » |
| F8 ★ | **Hanches trop basses / trop hautes** | `PLACEMENT` | Hauteur de hanche vs bande du modèle (§5.4) | Hors bande de 4 / 7 / 10 cm | **Au sol, avant la rep suivante** (« hanches un peu plus haut ») ; boucle fermée seulement pendant l'apprentissage à la barre vide |
| F9 | **Épaules derrière la barre** | `PLACEMENT` | Articulation de l'épaule vs barre, axe avant | Derrière de 2 / 4 / 6 cm | Au sol, avant la rep suivante |
| F2 ★ | **Hanches qui montent avant les épaules** | Du décollage (daté) au passage des genoux | Variation mesurée de l'angle du tronc **moins** la variation prédite (§5.4) ; contrôle croisé : rapport des montées hanche / épaule | 10° / 15° / 20° ; rapport > 1,4 | Au sol |
| F3 ★ | **Barre qui s'éloigne du corps** | Tirage | Écart avant maximal du centre de barre vs sa position au décollage | 3 / 5 / 8 cm | Au sol |
| F4 | **Lockout incomplet** | `HAUT` | Déficit signé d'extension de hanche ou de genou vs la référence debout | 8° / 12° / 20° | Au sol |
| F5 | **Hyperextension** | `HAUT` | Tronc en arrière (signé) vs la référence debout | 8° / 12° / 18° | Au sol |
| F6 | **Asymétrie** | Tirage | Décalage latéral du bassin vs milieu des pieds ; inclinaison de la barre | 3 / 5 / 7 cm | Au sol |
| F7 | **Bras pliés** | Tirage | Flexion du coude vs la référence debout | 15° / 25° / 35° | Au sol |

★ = faute héros.

- **Couplage F8/F9** : la bande de hanche de F8 est calculée avec l'épaule au centre de sa bande (3 cm devant la barre). Si les deux fautes sont présentes, F9 est cuée en premier, car la hauteur de hanche suit la position des épaules.
- **Unité d'analyse** pour la précision des fautes de placement : un placement = une décision, même si le cue est répété en boucle fermée.
- **Prise mixte** : la rotation du tronc est retirée de F6, et le type de prise est enregistré (§9.3).

### 5.2 Fautes v1.1

Arrachage sans mise en tension ; descente non contrôlée ; genoux qui avancent trop tôt à la descente ; « hitching » ; position de la tête ; largeur de prise et de pieds ; genoux qui rentrent ; talons qui décollent ; perte de vitesse de la barre (fatigue, VBT).

### 5.3 Dos rond : proxys honnêtes

- **Ce que voit le modèle** : le modèle multi-caméra est **halpe26** (`multi_camera.py:131-138`). Il sort aussi la tête, le cou et le centre du bassin, mais `rtmpose.py:47-51` ne garde que 21 points (COCO 17 + pointes + talons). **Aucun point n'est sur la colonne lombaire** : on ne mesure pas le dos rond et on ne l'affirme jamais.
- **« Perte de position »** = F2 + F3 + F9.
- **Signaux expérimentaux**, évalués sur données mais pas cués en v1 :
  - raccourcissement épaule ↔ hanche par rapport à la longueur debout. Le bruit d'un keypoint triangulé est ≈ 1,6 cm à une confiance de 0,6 et ≈ 2,4 cm à 0,4 (`utils/segment_lengths.py:19-21`), donc comparable à l'effet : le signal doit être moyenné sur le tirage ;
  - **points cou et tête de halpe26**, pour le haut du dos et la position de la tête. Ils sont récupérés par une table de correspondance propre au deadlift, sans modifier `rtmpose.py`.
- Un signal n'est activé que si son AUC est ≥ 0,8 sur des lifters jamais vus.
- Les cues restent comportementaux, jamais médicaux. La `BackRoundingRule` actuelle n'est pas utilisée.

### 5.4 Modèle du placement attendu (cœur de l'IP)

**Contraintes** (sous forme de bandes) :
- barre au-dessus du milieu du pied ;
- articulation de l'épaule 0–6 cm devant la barre ;
- bras **quasi verticaux** (≤ 5° vers l'arrière, conséquence de l'épaule placée devant la barre) ;
- tibia en contact avec la barre.

**Entrées** :
- longueurs tibia et fémur **projetées dans le plan sagittal**, mesurées sur les frames de placement ;
- torse et pied (`SegmentLengthEstimator`) ;
- bras : estimateur deadlift (épaule → coude → poignet) ;
- décalage poignet → prise : 6–9 cm, appris pour chaque personne ;
- distance entre l'axe de la barre et la ligne cheville–genou au contact : 5 cm au départ ;
- positions de la barre et de la cheville. Tout est relatif à la cheville, sans le sol.

**Résolution 1 (position de départ)** :
- l'épaule est placée dans sa bande, la cheville est fixée ; il reste un seul degré de liberté ;
- le contact du tibia donne deux solutions : on garde le genou devant la cheville, avec une flexion dans [40°, 120°] ;
- s'il n'y a aucune solution, F8 et F9 sont désactivées pour la personne, et c'est journalisé.

**Résolution 2 (passage des genoux)** :
- barre à hauteur des genoux, tibias quasi verticaux, épaules au-dessus de la barre ⇒ angle du tronc prédit ;
- **variation prédite pour F2** = angle prédit au passage des genoux − angle prédit au départ.

**Validation** :
- la référence est la hauteur de hanche **mesurée** sur les placements annotés « bons » par le coach (marqueur sur le grand trochanter, caméra sagittale, précision ≤ 1 cm) ;
- si l'erreur du modèle au 95e percentile est ≤ 2,3 cm, F8 est cuée au niveau modéré ;
- si elle est ≤ 3,3 cm, F8 n'est cuée qu'au niveau sévère ;
- au-delà, F8 est désactivée et la démo passe au plan B.

### 5.5 Personnalisation

- F4, F5 et F7 sont relatives à la position debout ; F1, F3 et F6 sont absolues ; F2, F8 et F9 sont relatives au modèle.
- Seuil = max(seuil de base, 3 × bruit mesuré à l'échauffement), avec un plafond. On n'utilise jamais « pic observé + marge ».
- Stockage sous `hip_hinge` avec le marqueur `dl_v1` et des seuils sous clés `dl_*`.
- Mesures corporelles de départ : celles de la ligne `dl_v1`, sinon les `athlete_params` du squat en lecture seule (§3.3). Elles sont ensuite remesurées pendant la séance.

### 5.6 Budget d'erreur

Règle : on ne cue un niveau que si l'erreur au 95e percentile (mesurée contre la vérité terrain, §8.4) est ≤ 1/3 du seuil de ce niveau. Sinon, on ne cue qu'à partir du niveau supérieur.

| Faute | Seuil léger | Erreur max pour cuer « léger » | Erreur attendue (à vérifier) | Niveau cué au départ |
|---|---|---|---|---|
| F1 | 3 cm | 1 cm | ≤ 1 cm (statique, ≥ 15 frames) | léger |
| F3 | 3 cm | 1 cm | 1–2 cm | modéré tant que non prouvé |
| F6 | 3 cm | 1 cm | 1–2 cm | modéré |
| F2 | 10° | 3,3° | 2–4° + erreur du modèle | modéré tant que non prouvé |
| F4 / F5 | 8° | 2,7° | 2–3° (relatif) | modéré si la vérité terrain dépasse 1,5° |
| F7 | 15° | 5° | 3–5° | léger |
| F8 | 4 cm | 1,3 cm | voir §5.4 | modéré ou sévère |
| F9 | 2 cm | 0,7 cm | 1–1,5 cm | modéré |

## 6. Barre au sol : détection et suivi 3D

### 6.1 Existant

- `BarbellDetector` est un YOLO11n-pose qui ne renvoie que la meilleure détection (`:113`). Ses poids ne sont pas dans le repo, et on ignore sur quelles données il a été entraîné.
- Il est désactivé par défaut (`config.py:145`). En multi-caméra, il ne tourne qu'en 2D, sur la caméra principale (`pipeline.py:613-624` ; `multi_camera.py:307-357`).
- La barre n'est triangulée que pour l'échelle de la calibration (`person_calibration.py:599-601, 852, 968`).
- `BarPathRule` mélange les axes ; on ne la réutilise pas.

### 6.2 Keypoints

- Barre chargée : centre de la face extérieure du disque extérieur, de chaque côté.
- Barre vide : extrémités du manchon, dans une classe distincte.
- Choix à confirmer sur les premières images réelles (J1).

### 6.3 Données, entraînement, licence

1. Récupérer les poids actuels et les évaluer tels quels (J1).
2. Extraire des images des enregistrements bruts (§8.2) : 3 caméras, toutes les phases, négatifs (barre rangée dans les J-hooks), plusieurs types de disques, couleurs et éclairages.
3. Annoter 2 000 à 3 000 images dans CVAT auto-hébergé.
4. Entraîner, puis exporter en ONNX → TensorRT FP16.
5. Évaluer par lifter : rappel ≥ 95 % au sol, faux positifs ≤ 1 % sur la barre rangée.
6. **Licence** : Ultralytics (YOLO11) est sous **AGPL-3.0**. Embarquer la bibliothèque ou des poids qui en dérivent dans un appareil vendu demande une licence Entreprise Ultralytics, ou une alternative sous licence permissive (par exemple RTMDet / RTMPose de MMPose, Apache-2.0, déjà dans la famille du pipeline de pose). Décision d'Ambaka **avant J4** ; elle concerne aussi le détecteur de barre existant.

### 6.4 Exécution pendant une série

1. `get_frames_and_pose()` fournit les images des 3 caméras (§3.1).
2. `bar_detector.py` (nouvelle classe) traite les 3 vues en lot et renvoie **toutes** les détections.
3. On corrige la distorsion (`calibration.undistort_keypoints`).
4. Association multi-vues : erreur de reprojection minimale, cohérente avec les mains et les pieds. La barre rangée est éliminée à cette étape.
5. Gauche et droite sont distingués grâce à la ligne des épaules.
6. DLT sur les deux extrémités (≈ 40 lignes, car les fonctions de `triangulator.py` sont privées), avec rejet si la reprojection est mauvaise.
7. Un filtre de Kalman 3D par extrémité.
8. Sortie `BarState3D` dans le repère lifter : centre, hauteur relative, avant/arrière par rapport au milieu du pied, inclinaison, vitesse verticale, `source`, confiance.

### 6.5 Milieu du pied et sol

`foot_contact.py` n'est pas touché.

- Les ancres des pieds sont verrouillées en `APPROCHE`/`POSITION`, en moyenne sur ≥ 15 frames.
- Le **milieu du pied est verrouillé avant le tirage**, donc il ne dépend pas des occultations pendant le tirage.
- Les métriques sont relatives (§2.2). Le sol absolu ne sert qu'à l'affichage et aux contrôles :
  - si le diamètre des disques est connu (§9.3) : barre au repos moins le rayon du disque ;
  - sinon : ancres des pieds moins un décalage estimé à chaque séance.
- Tout est remis à zéro quand la calibration change.

### 6.6 Repli sans barre

On utilise le milieu des poignets, moins le décalage appris. F1, F3 et l'inclinaison de F6 passent en `source = wrist_proxy` : seuils élargis, cue seulement à partir de « modéré ». Si la confiance est trop basse, la faute n'est pas émise.

### 6.7 Coût edge

Mesuré sur Jetson dès J1. Leviers, dans cet ordre :
1. TensorRT FP16 ;
2. images en 480 px ;
3. détection 1 frame sur 2, avec le Kalman entre les deux ;
4. recadrage autour des mains et des pieds ;
5. seulement les 2 caméras à 45°.

## 7. Diagnostic (graphe statique)

### 7.1 Principe

- Même méthode que le squat : symptômes, causes, a priori écrit à la main × preuve, fuite, noisy-OR. Même schéma `DiagnosisResult`.
- Module séparé `deadlift/diagnosis/`, car le moteur squat charge ses graphes en variables globales (`diagnosis/graph/loader.py:114-116`).
- Sorties simples : `cause_id`, niveau, score, **un** delta chiffré, une phrase.
- **Diagnostic glissant** : recalculé après chaque rep (< 1 ms) et envoyé en `dl_diagnosis_update`, pour que le récap ait toujours le diagnostic de la série en cours (§9.2). Le `diagnosis_complete` final part en fin de série, pour la base de données.

### 7.2 Symptômes v1

`dl_bar_not_over_midfoot` (F1), `dl_setup_hip_height` (F8), `dl_shoulders_behind_bar` (F9), `dl_hips_rise_early` (F2), `dl_bar_drift` (F3), `dl_incomplete_lockout` (F4), `dl_overextension` (F5), `dl_lateral_shift` (F6), `dl_bent_arms` (F7).

### 7.3 Causes v1

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
| `dl_anthropometric_context` | 0 | — | fémurs longs / bras courts |

### 7.4 Score de rep

Placement (F1, F8, F9) 25 % ; coordination (F2) 25 % ; trajectoire de barre (F3) 20 % ; lockout (F4, F5) 15 % ; symétrie (F6) 15 %.

### 7.5 Fantôme

Option pour la démo : le placement idéal du §5.4 affiché comme fantôme statique.

## 8. Données

### 8.1 Connaissance statique

`docs/deadlift/KNOWLEDGE.md` contient :
- les phases ;
- les métriques, avec leur repère, leur signe et leur phase ;
- les seuils et leur justification ;
- les cues ;
- le modèle géométrique.

Sources : coaching classique et manuels de préparation physique. Revue par Ambaka, et si possible par un coach de force externe.

### 8.2 Enregistreur brut et rejeu (J1, prérequis)

- **`scripts/tools/record_rig.py`** : outil autonome (via `multi_capture`). Il enregistre une vidéo par caméra, les horodatages, une **copie** des fichiers de calibration et de gravité, et les métadonnées de séance (§9.3). Il n'écrit rien dans `~/.nowva`.
- **`deadlift/replay_provider.py`** : même interface que le fournisseur de caméras. Il sert à faire tourner le deadlift hors ligne et aux tests de rejeu squat (§3.4.7).
- Volume : ≈ 5 Go pour 10 min, stockage local chiffré.

### 8.3 Données réelles et plan de test

**Round 1 (équipe, semaines 1–2)** :
- 2 à 3 personnes, barre vide et charges légères ;
- reps propres et fautes volontaires sans danger ; dos rond uniquement à la barre vide ou au bâton ;
- sert aussi à mesurer la **corrélation intra-lifter (ρ)** de chaque faute.

**Round 2 (pilote)** :
- lifters de 1,55 à 1,95 m, morphologies variées, dont **≥ 4 débutants** (F2 et F3 y sont fréquentes) ;
- séries naturelles en **mode observation** (cues coupés) et séries scénarisées.

**Jeu de test, dimensionné avec l'effet de grappe.** Les reps d'un même lifter sont corrélées : l'effet de plan vaut 1 + (m − 1)·ρ, où m est le nombre de cues par lifter. Hypothèse de départ ρ = 0,1, recalculée avec le ρ mesuré au round 1.
- **≥ 10 lifters de test, avec ≥ 15 positifs par faute héros et par lifter** (soit ≥ 150 positifs par faute héros). Avec un rappel de 0,7 et une précision de 0,85, la borne basse effective est ≈ 0,74 pour la précision (porte démo : 0,70) et ≈ 0,58 pour le rappel (porte : 0,55).
- La porte de lancement (borne basse ≥ 0,75) demandera ≈ 12 lifters × 20 positifs, à recalculer avec le ρ mesuré.
- Fautes provisoires : ≥ 5 positifs par lifter (≥ 50 au total) ; intervalles rapportés tels quels.
- **≥ 30 reps propres par lifter** (≥ 300 au total), pour le taux de fausses corrections.
- Volume total : 10 × (60 + 25 + 30) ≈ 1 150 reps, soit ≈ 23 séries de 5 reps par lifter, en 2 séances à charge légère.

**Statistiques de la porte J6** :
- bootstrap par lifter, complété par un bootstrap-t et une validation « un lifter laissé de côté » ;
- plancher par lifter (≥ 0,70), appliqué seulement aux lifters qui ont ≥ 15 cues de la faute ;
- séries scénarisées et naturelles rapportées séparément : une faute héros doit atteindre ≥ 0,80 sur ≥ 30 cas naturels, sinon elle reste provisoire ;
- **plan B accepté d'avance** : si F2 n'atteint pas ses cas naturels, la démo se fait avec F1 + F3 (décision à J6).

**Annotation** (CVAT, 3 vues) :
- fautes et niveaux pour chaque rep ;
- événements horodatés sur un sous-ensemble ;
- double annotation de 20 % des reps : κ ≥ 0,6 exigé pour chaque faute ; arbitrage par Ambaka ; une faute sous ce κ est redéfinie.

**Entraînement et réglage** : ≥ 300 reps et ≥ 50 positifs par faute, en plus du jeu de test ; 2 000–3 000 images de barre.

**Éthique** : consentement écrit, anonymisation, stockage local.

### 8.4 Vérité terrain

| Grandeur | Méthode | Précision visée |
|---|---|---|
| Barre statique / F1 | Scotch au sol et gabarit de pied (0 / 3 / 6 / 10 cm) | ≤ 3 mm |
| Barre dynamique (F3, vitesse) | Marqueurs ArUco au moyeu des disques, triangulés ; en option, un capteur de position linéaire | ≤ 5 mm |
| Angles tronc / hanche / genou (F2, F4, F5) | **Référence : caméra sagittale plane calibrée** (damier dans le plan, 120 fps), avec marqueurs sur le grand trochanter et l'acromion. C'est la même ligne que la métrique vision. Les centrales inertielles (dos, cuisse) ne servent qu'au chronométrage et aux contrôles : celle du dos mesure le segment thoracique, et celle de la cuisse bouge avec la peau | ≈ 1–1,5° |
| Hauteur de hanche au placement (F8, §5.4) | Marqueur sur le grand trochanter, caméra sagittale | ≤ 1 cm |
| Gravité | Planche à plat + niveau à bulle | ≤ 0,3° |
| Événements | Horodatages annotés | 1 frame |

### 8.5 Données synthétiques (simulateur)

On étend `.claude/preik-audit/harness/preik_harness/` :
- corriger les chemins en dur (`__init__.py:13`) et rendre `runner.py:404` paramétrable ;
- écrire un générateur deadlift : départ au sol, mains sur la barre, barre au sol (aujourd'hui, `barbell.py` la place sur le dos), occultation par les disques, monde incliné, caméra déplacée ;
- scénarios : F1–F9, lockouts mous comptés, rep bloquée, arrêt complet, relance rapide, touch-and-go (séries de 5 et 10 reps), rebond de bumper, barre lâchée, re-placement, plusieurs morphologies. **Assertion : 100 % des reps comptées exactement une fois, touch-and-go compris, y compris avec un tracker bruité (bruit de vitesse ×3) pour la dernière rep d'une série.**

Ces données ne servent **jamais** à fixer les seuils finaux.

### 8.6 Questions auxquelles les données doivent répondre avant la démo

1. Qualité de la pose en position penchée.
2. Occultations, caméra par caméra.
3. Erreur réelle de chaque métrique et du modèle de placement.
4. Bruit sur les reps propres, marge de lockout, décalage du décollage.
5. ρ intra-lifter, précision et rappel.
6. Signaux expérimentaux sur le dos : gardés ou abandonnés.

## 9. Coaching vocal, cycle de vie et messages

### 9.1 Parcours utilisateur

**Entrées possibles**
- Exercice rapide (« on fait du deadlift »), ou séance programmée composée **uniquement** de deadlifts.
- Le sous-processus est lancé une fois par séance, avec le premier exercice (`main.py:1065-1099`). **En v1, une séance programmée n'est prise en charge pour le deadlift que si tous ses exercices sont des deadlifts.** Si une séance mélange les exercices et commence par un deadlift, Nova propose de faire le deadlift en séance rapide, et le programme reste inchangé. Une séance qui commence par un squat est inchangée.
- Changer d'exercice en cours de séance (redémarrage du sous-processus) : v1.1, avec l'accord d'Ambaka.
- Sumo, RDL et autres variantes sont refusés par le code.

**Première fois** (pas de ligne `dl_v1`), en parcours rapide **comme** en séance programmée :
- En séance programmée, la branche deadlift de `start_workout` passe d'abord par l'apprentissage. Sinon le `WorkoutAgent` (`main_menu_agent.py:119` → `workout_agent.py:146`) activerait la séance tout de suite, et les reps d'apprentissage seraient enregistrées comme des reps de séance.
- L'apprentissage (textes séparés) couvre :
  1. pieds à largeur de hanches, barre au-dessus du milieu du pied ;
  2. prise juste à l'extérieur des jambes ;
  3. tibias contre la barre ;
  4. épaules au-dessus de la barre, dos plat ;
  5. mise en tension, puis pousser le sol.
- Le **type de prise** et les **disques** sont demandés **avant** l'ouverture des caméras (§9.3).
- Quand le pipeline suit le lifter et la barre, il envoie `assessment_ready` avec `exercise`. Le callback existant (`coaching_service.py:609-620`) lance alors « vas-y, première rep ».
- Suivent 2–3 reps à la barre vide, avec un guidage en boucle fermée de F1, F8 et F9 (sans danger à vide), puis la mesure du bruit.
- La phase se termine par `calibration_complete` (`dl_v1`), qui active la séance (`coaching_service.py:963-968`).

**Utilisateur connu** : passage direct par le `WorkoutAgent`, qui active la séance à l'entrée.

**Calibration caméra** : si le deadlift n'a pas encore sa copie, elle est amorcée depuis celle du squat (§3.1). S'il n'existe aucune calibration, la calibration à partir du lifter demande 2 squats lents au poids du corps (l'agent l'annonce) et le résultat est écrit dans le dossier deadlift.

**Guidage du pied en boucle fermée** (moment fort de la démo), en `POSITION` : un cue directionnel F1 toutes les 1,5 s tant que l'écart persiste, puis « parfait ». C'est le même principe que le moniteur d'ajustement squat (`coaching_orchestrator.py:381-520`), sans y toucher.

**Politique de cues pendant une série** (décidée par le pipeline deadlift, seule source de vérité) :
- jamais de correction pendant le tirage ;
- au plus 1 correction par rep, envoyée quand la barre est au sol ;
- priorité : F1 → F9 → F8 → F2/F3 → F4/F5 → F6 → F7 ;
- rien entre deux reps en touch-and-go ;
- ces cues ne passent pas par `orchestrator.on_fault`, dont l'écart minimal de 8 s (`:152`) en supprimerait une sur deux.

**Cues perdus** : `_dispatch_cached_cue` abandonne les cues de plus d'1 s (`:966`), et un cue attend que le LLM ait fini de parler. On ne modifie pas cette règle partagée.
- La boucle fermée renvoie le cue toutes les 1,5 s, donc le guidage reprend dès que Nova se tait.
- Le plafond de 4 cues par placement est tenu **côté voix** et ne compte que les cues **réellement joués**, signalés par `on_fault_cue_delivered`.
- Les cues de placement utilisent des types `dl_setup_*`, jamais utilisés pour un `CueEvent`. `_ops_cue_delivered` (`biomechanics_persistence.py:631-636`) ne trouve donc rien et ne modifie aucune ligne.
- Un même cue n'est jamais répété à moins de 3 s ; chaque abandon est journalisé et compté.

**Audio**
- Cues directionnels à deux intensités. Les clips sont pré-générés dans `wav_deadlift/` et chargés seulement en séance deadlift.
- **Aucune synthèse de clip pendant la séance** : à J5, on vérifie qu'aucun appel à `generate_tts` / `_generate_single_cue` n'a lieu pendant une séance deadlift (tous les clips existent).
- Les chiffres (« 4 cm ») sont dits par le LLM conversationnel dans le récap.
- **Honnêteté sur la pile vocale** : aujourd'hui, la couche conversationnelle est **dans le cloud** (Deepgram STT, LLM OpenAI `gpt-5.4-mini` par défaut, Cartesia via LiveKit Inference ; `voice_agent.py:132-161`), ce qui va contre le garde-fou n°1. Le deadlift n'ajoute **aucune** dépendance cloud et garde des sorties à la taille d'un petit modèle local, prêtes pour la migration. Cette migration ne fait pas partie de ce plan (§14).

**Récap**
- Déclenché par le compte de reps de l'orchestrateur, comme pour le squat.
- Il lit le dernier `dl_diagnosis_update`, déjà reçu à ce moment-là. Il n'y a donc pas de course avec `_consume_diagnosis` (`coaching_orchestrator.py:369-375`), qui n'attend pas.
- Le delta chiffré est formaté par la branche deadlift (§3.3).

### 9.2 Contrat de messages

**Pipeline deadlift → `main.py` → voix.** Tous les messages portent `exercise = "conventional_deadlift"`. Les noms de champs sont ceux que le code lit déjà.

| Type | Émis quand | Champs | Consommateurs |
|---|---|---|---|
| `pipeline_status` | Démarrage, préchargement | `status` | `main.py:973` (inchangé) |
| `cache_cues` | Début de séance | `cues` (clés deadlift) | `coaching_service._on_cache_cues` → `audio_cue_service.load_extra_cues` (branche deadlift) |
| `assessment_ready` | Lifter et barre suivis, en apprentissage | `exercise` | Transfert **ajouté** si `exercise` est présent → `coaching_service:609` |
| `assessment_rep`, `assessment_result`, `calibration_rep` | Apprentissage | Mêmes champs que le squat | Transfert existant (`main.py:1035`) |
| `calibration_complete` | Fin de l'apprentissage | `movement_pattern = "hip_hinge"` (**explicite** : le défaut est `"squat"`, `coaching_service.py:915`), `peaks`, `thresholds` (`schema: "dl_v1"`, clés `dl_*`), `athlete_params`, `baseline` | Sauvegarde et activation (~915-968) |
| `dl_cue` (nouveau) | Guidage, et correction après une rep | `cue`, `kind` (`setup_correction` / `setup_confirm` / `rep_correction`), `placement_id`, `fault_type` (`dl_setup_*` ou `dl_*`), `severity`, `severity_score`, `rep_number` (rep courante), `message`, `value`, `target`, `source` | `coaching_service` → `orchestrator.play_cached_cue`. Pour `rep_correction`, aussi `recorder.record_fault(message)`, avec accusé de livraison automatique |
| `dl_diagnosis_update` (nouveau) | Après chaque rep, avant `rep_complete` | `diagnosis`, `scoring`, `set_number`, `reps_so_far` | `coaching_service` → `orchestrator.set_diagnosis_data` (méthode publique existante). Pas d'écriture en base |
| `rep_complete` | À la transition de comptage (§4.4) : arrêt au sol (`DESCENTE → AU SOL`) ou relance en touch-and-go (`DESCENTE → TIRAGE`), après `dl_cue` et `dl_diagnosis_update` | `rep_number`, `set_number`, `is_clean`, `faults_in_rep`, `faults_detailed` (toutes les fautes : {`fault_type`, `severity`, `severity_score`}), `rep_duration_ms`, `ascent_time_s` (= durée du tirage), `descent_time_s`, `depth_category = ""`, `max_depth_angle = 0`, `rep_kinematic_summary` (`dl_schema: 1`), `bar_source` | `coaching_service` → `on_rep_complete` (champs de profondeur neutres) ; enregistreur (`_build_rep_row`, inchangé) ; affichage |
| `diagnosis_complete` | Fin de série (`rest_start` reçu, ou `workout_complete`) | `diagnosis`, `scoring` {`mean_score`, `per_dimension`, `best_rep`, `worst_rep`}, `set_number` | `coaching_service` → `record_set` (base) et `set_diagnosis_data` |
| `set_complete` | Fin de série | Comme le squat | Affichage seulement (`main.py:1005`) ; pas transféré à la voix |
| `rest_complete`, `frame_data` | Comme le squat | `rep_phase` deadlift | Inchangé |
| `dl_status` (nouveau) | Problème qui demande une action de l'utilisateur | `code` (`camera_moved`, `gravity_missing`, `bar_not_tracked`), `message` | `main.py` (ajout à la liste de transfert) → `coaching_service` (branche deadlift) : courte phrase de Nova à partir d'un clip pré-généré, plus un bandeau à l'écran |
| Réponses aux demandes de rejeu | `request_last_rep` / `request_demo` | `last_rep_snapshot {request_id, error: "no_data"}` / `demo_data_ready {request_id, status: "unavailable"}`, comme le squat (`pipeline_process.py:1720-1756`) | Libère `_request_from_pipeline` (`coaching_service.py:340-369`) immédiatement |

**Voix → pipeline deadlift.** Ce sont les messages existants filtrés par `main.py:601-603`, tous gérés par `deadlift/process.py` :
- `rest_start` : finalise la série (`diagnosis_complete`, `set_complete`), lance le minuteur de repos et réarme la porte d'approche ;
- `workout_complete` : voir §9.3 ;
- `assessment_mode` : bascule apprentissage / séance ;
- `request_last_rep`, `request_demo` : réponses ci-dessus ;
- `demo_*` : ignorés.

### 9.3 Démarrage, arrêt, métadonnées

- **Arguments et poignée de main** : `deadlift/process.py` accepte exactement les arguments que `main.py` passe à `pipeline_process.py` (`cam0 cam1 exercise`, fichier de calibration, mode calibration, préchargement ; `main.py:1088-1125`). Il reproduit la poignée de main du préchargement : chargement des modèles, attente de `start_capture` (`pipeline_process.py:895-905`), puis ouverture des caméras.
- **Exercice connu dès l'activation** : la voix lit `workout.exercise_name` dans l'état (reconnu par `names.py`) au moment où l'enregistreur est créé (§3.3). Elle n'attend pas un premier message portant `exercise`.
- **`workout_complete`** : finalise la série en cours (`diagnosis_complete`, `set_complete`), vide les messages, ferme les caméras et sort proprement, comme le bloc `finally` de `pipeline_process.py:1965-2043`.
- **Métadonnées de séance** : type de prise (double, mixte ou crochet), diamètre des disques (45 cm par défaut), ceinture, chaussures. Elles doivent arriver **avant l'ouverture des caméras**, car le sous-processus est lancé dès le passage en mode séance (`main.py:918-1099`) :
  - **parcours rapide** : demandées par la branche deadlift de `CollectExerciseInfoTask`, avant le changement de mode ;
  - **séance programmée** : demandées pendant l'accueil, avant que `workout.greeting_done` soit posé. Pour un utilisateur connu, par la branche deadlift du `WorkoutAgent` (`greeting_done` posé à `workout_agent.py:194`). Pour une première fois, par l'agent d'apprentissage deadlift (le pendant de `teaching_agent.py:122`, où l'agent squat pose `greeting_done`) ;
  - la branche deadlift de `main.py` joint ces valeurs, lues dans l'état, au message `start_capture` envoyé après `greeting_done` (`main.py:1104-1118`) ; `deadlift/process.py` les lit à la réception de ce message ;
  - si l'utilisateur ne sait pas : prise double, disques de 45 cm ;
  - pas de changement en cours de séance en v1.

## 10. Edge (Jetson)

- **Mesure dès J1** sur Jetson Orin Nano Super : RTMPose sur 3 vues + YOLO11n-pose (ou son alternative, §6.3) sur 3 vues, en TensorRT FP16, avec la chaîne complète.
- Les nouveaux calculs coûtent < 1 ms.
- **Mode dégradé**, dans l'ordre :
  1. pose à 30 Hz, barre à 15 Hz avec Kalman ;
  2. sinon, tout à 15 Hz (à vérifier pour le chronométrage du décollage).
- Le deadlift n'ajoute aucune dépendance cloud et ne synthétise aucun clip en séance. La couche conversationnelle existante est encore dans le cloud (§9.1) ; la migrer vers un modèle local est hors de ce plan. Annotation et entraînement se font hors ligne.

## 11. Jalons, critères d'acceptation, estimations

Estimations pour 1 ingénieur à temps plein. À 2, J2 et J4 peuvent avancer en parallèle, de même que J3 et J4.

| Jalon | Contenu | Critère d'acceptation | Dépend de | Estim. |
|---|---|---|---|---|
| **J0 — Filet squat** | `requirements.lock` ; golden masters pipeline et voix ; prompts ; alias ; manifeste de gel ; test « aucune écriture squat » (avec affinage forcé) ; drapeau sans effet ; `.claude/rules/deadlift.md` ; CI (si validée) ; décision sur la licence YOLO | Tout vert sur `main`. Une modification volontaire d'un seuil squat ou d'un texte de cue squat fait échouer le bon test | — | 4–6 j |
| **J1 — Fondations** | Enregistreur + rejeu ; outil de gravité + repère lifter ; `KNOWLEDGE.md` ; modèle de placement (2 résolutions) ; capture round 1 (avec mesure de ρ) ; mesure Jetson ; évaluation des poids existants | Revue signée ; tests de signe, d'inclinaison, de cohérence et test statique verts ; ≥ 1 h de capture brute ; keypoints de barre décidés ; chiffres Jetson | J0 | 2 sem. |
| **J2 — Simulateur deadlift** | Générateur, scénarios | Chaque scénario produit sa vérité terrain | J1 | 1 sem. |
| **J3 — Cœur deadlift + squelette de bout en bout** | Fournisseur caméra ; `DeadliftPipeline` (contrats) ; calibration caméra séparée ; machine à états (arrêt complet, relance, décollage daté) ; métriques F1–F9 ; golden master deadlift. **Tranche fine de bout en bout, derrière le drapeau** : bifurcation dans `main.py`, processus deadlift, `rep_complete` → la voix compte les reps sur le rack | Simulateur : 100 % des reps comptées exactement une fois (arrêt au sol, relance, touch-and-go, rebond de bumper), événements ≤ 100 ms, chaque faute injectée détectée, scénario propre sans faute. Sur le rack : « deadlift » → reps comptées à voix haute | J2 | 2,5 sem. |
| **J4 — Barre 3D** | Annotation, entraînement, export ; tracker 3D ; association multi-vues ; vérité terrain ArUco | Rappel ≥ 95 % ; erreur 3D statique ≤ 1 cm et dynamique ≤ 1,5 cm ; budget Jetson tenu ou mode dégradé validé | J1, décision de licence | 2–3 sem. |
| **J5 — Intégration complète** | Contrat §9.2–9.3 ; fichiers du §3.3 ; audio deadlift ; affichage ; base de données ; guidage ; récap | Séance complète en parcours rapide et programmé (première fois et utilisateur connu) ; liaison cue ↔ rep en base ; provenance ; isolation des données ; zéro synthèse de clip en séance ; affinage deadlift rechargé à la séance suivante ; aucune écriture squat ; golden masters squat identiques, drapeau éteint et allumé | J0, J3, J4 | 2 sem. |
| **J6 — Validation réelle** | Round 1 → seuils, budget d'erreur, marges, ρ ; round 2 → jeu de test | Porte démo du §1 sur ≥ 10 lifters (effet de grappe, bootstrap par lifter, séries naturelles à part) ; κ ≥ 0,6 ; statut fixé pour chaque faute | J5 | 3–4 sem. |
| **J7 — Diagnostic deadlift** | Graphe statique, score, récap chiffré | La cause n°1 correspond à l'annotation du coach dans ≥ 70 % des séries | J6 | 1–1,5 sem. |
| **J8 — Démo** | Scénario héros : boucle fermée F1, puis correction de F8, F2 ou F3 à la rep suivante. Plan B : F1 + F3 | 10 démos consécutives sans erreur ; budget Jetson tenu | J7 | 1 sem. |

Total : ≈ 17–19 semaines pour 1 personne, ≈ 10–11 semaines pour 2. La capture réelle commence dès J1, et une première démo vocale de bout en bout existe dès J3.

## 12. Risques et parades

| Risque | Parade |
|---|---|
| Toucher au squat par accident | Sous-processus séparé, composition, drapeau, golden masters pipeline et voix, manifeste de gel, CI |
| Écrire un état squat (calibration caméra, base) | Calibration caméra copiée par paire dans `~/.nowva/deadlift/`, ré-amorcée si le squat recalibre ; outil de gravité dédié ; test « aucune écriture » avec affinage forcé ; isolation des données |
| Vertical faux / caméra déplacée | Gravité mesurée par caméra + contrôle de cohérence ; tests d'inclinaison ; repli à seuils élargis |
| Règle `.claude` fausse sur les axes | `.claude/rules/deadlift.md` + test statique sur Y/Z |
| Dos rond non mesurable | Proxys, vocabulaire comportemental, signaux expérimentaux soumis aux données |
| Pose dégradée en position penchée, occultations | Mesures dès le round 1 ; milieu du pied verrouillé ; affinage RTMPose hors v1 |
| Détecteur de barre qui ne généralise pas | Données variées, découpage par lifter, négatifs |
| Licence AGPL d'Ultralytics | Décision avant J4 : licence Entreprise ou alternative Apache-2.0 |
| Seuils inventés | Valeurs initiales seulement, budget d'erreur, données réelles |
| Validation trop optimiste | Effet de grappe, bootstrap par lifter, séries naturelles à part, plancher par lifter |
| Récap sans son diagnostic | Diagnostic glissant |
| Cues perdus pendant que Nova parle | Boucle fermée qui renvoie le cue, plafond compté sur les cues joués |
| Ancienne ligne `hip_hinge` de forme squat | Marqueur `dl_v1`, tests |
| Budget Jetson | Mesure à J1, mode dégradé |
| Duplication qui diverge | Golden master deadlift, tests de contrats |
| Prise mixte | Rotation retirée de F6, type de prise enregistré |

## 13. Hors périmètre v1

Sumo, RDL et autres variantes ; changement d'exercice en cours de séance ; séances programmées mixtes ; migration de la couche conversationnelle vers un modèle local ; apprentissage des probabilités ou du LLM ; fantôme animé ; ML pour le dos rond ; VBT complet ; rejeu de la dernière rep deadlift ; toute modification d'un fichier squat.

## 14. Questions pour Ambaka

1. Qui annote ?
2. Le touch-and-go est-il compté en v1 (proposition : oui, sans correction entre les reps) ?
3. Où sont les poids actuels du détecteur de barre, et sur quelles données ont-ils été entraînés ?
4. Licence YOLO : licence Entreprise Ultralytics ou alternative Apache-2.0 ?
5. OK pour la CI GitHub Actions ?
6. OK pour la duplication assumée et pour une calibration caméra séparée pour le deadlift ?
7. Un accéléromètre dans le boîtier est-il envisageable ?
8. Quels disques dans la salle de démo ?
9. OK pour corriger la ligne fausse sur l'axe Y dans `.claude/rules/biomechanics.md` (documentation seulement) ?
10. Observations hors périmètre, côté squat (on n'y touche pas) :
    - l'axe vertical du monde n'est pas la gravité (§2.1) ;
    - `main.py:1035` ne transfère pas `assessment_ready` ni `shallow_rep`, alors que la voix les gère (`coaching_service.py:524, 609`) ;
    - drapeau éteint, un programme « Barbell Conventional Deadlift » n'est pas normalisé, et sa calibration est écrite sous la ligne **squat** (`pipeline_process.py:1073`) ;
    - le récap squat peut partir avant son `diagnosis_complete` (`_consume_diagnosis` n'attend pas, `coaching_orchestrator.py:369-375`) ;
    - **la couche conversationnelle est dans le cloud** (Deepgram, OpenAI, Cartesia via LiveKit Inference ; `voice_agent.py:132-161`), contrairement au garde-fou n°1 de `CLAUDE.md`.
