# Plan — Soulevé de terre conventionnel (deadlift) — v1

Statut : proposition, à valider par Ambaka. Version 2 du document.

## 0. Décisions déjà prises (ne pas rediscuter)

| Décision | Conséquence pour le plan |
|---|---|
| On sort du « squat only » de `CLAUDE.md` pour le deadlift | Chantier autorisé, mais il ne doit rien coûter au squat |
| **Le squat ne doit jamais être modifié et reste séparé du deadlift** | Aucun fichier squat n'est modifié ; le deadlift a son propre sous-processus et son propre package ; les fichiers partagés (voix, affichage) ne reçoivent que des ajouts derrière un drapeau ; deux filets de tests (pipeline et voix) prouvent que le squat ne bouge pas |
| Deadlift **conventionnel uniquement** | « Sumo » et toute autre variante sont refusés **dans le code**, pas seulement dans le prompt |
| Dos rond : proxys indirects acceptés | On ne prétend jamais mesurer la colonne |
| Les caméras du rack voient la barre au sol | Détection et suivi 3D de la barre au sol à construire |
| Apprentissage du LLM : hors périmètre | On enregistre seulement les données, étiquetées par exercice |
| **Piège des axes** | §2 — règle bloquante |

## 1. Objectif produit et critères de succès

Boucle démo YC deadlift :

> « On fait du deadlift » → Nova guide le placement **en boucle fermée** (« avance un peu… encore… parfait ») → l'athlète tire → le système détecte une faute mesurable → **un** cue au bon moment (au sol, entre deux reps) → rep suivante corrigée → récap de série.

| Métrique (données réelles, lifters jamais vus) | Cible |
|---|---|
| Comptage des reps | ≥ 99 %, 0 rep fantôme sur séries propres |
| Événements (décollage, passage genoux, lockout, sol) | erreur médiane ≤ 100 ms |
| Précision par faute émise en cue | ≥ 0,85 (borne basse de l'intervalle de Wilson à 95 % ≥ 0,75) |
| Rappel par faute émise en cue | ≥ 0,70 |
| Fausses corrections sur reps propres | ≤ 1 pour 10 reps |
| Erreur de mesure | ≤ 1/3 du seuil « léger » de chaque faute cuée à ce niveau (§5.6) |
| Latence d'un cue | identique au squat (audio pré-enregistré) |
| Squat | golden masters pipeline **et** voix identiques |
| Edge | ≤ 33 ms/frame sur Jetson Orin Nano Super, ou mode dégradé défini (§10) |

## 2. Repères et piège des axes (règle bloquante)

### 2.1 Ce que dit le code

- Repère monde : mètres, **Y vers le bas** (Y plus grand = plus bas), X = gauche du sujet, **+Z = dos du sujet** (avant = −Z). Sources : `triangulation/person_calibration.py:5`, `triangulator.py:2`, `utils/foot_contact.py:32`, `kinematics/analytical_ik.py:449`.
- `.claude/rules/biomechanics.md` (« Larger Y = higher ») et `README.md:325` (« Z-forward ») sont **faux** par rapport au code.
- **L'axe Y du monde n'est pas la gravité.** `world_frame_from_standing` (`person_calibration.py:734-752`) prend Y = direction médiane hanche → cheville debout, ce qui peut s'écarter de la verticale de 2 à 4°. Sur 55 cm de course de barre, 3° d'écart créent ≈ 2,9 cm de fausse dérive avant/arrière : c'est le seuil de F3.

### 2.2 Repère du lifter (utilisé par toutes les métriques deadlift)

Aucune métrique deadlift n'utilise directement Y ou Z bruts. Tout passe par `src/biomechanics/deadlift/frame.py` :

1. **Vertical = normale au sol (gravité)**, dans cet ordre de préférence :
   - (a) **Rack produit** : les caméras sont fixées rigidement au rack. La gravité est mesurée une fois en usine avec la planche ChArUco posée à plat au sol (`scripts/tools/calibrate_cameras.py --flat-placement`, `triangulation/charuco.py:476-481`). On la stocke **dans le repère de la caméra 0**, dans un fichier appartenant au deadlift (`~/.nowva/deadlift_gravity_<camera_key>.json`). On la ramène dans le repère monde courant avec les rotations des fichiers de calibration (lecture seule), même après le recentrage sur le lifter (`_refined`).
   - (b) **Rig de dev (trépieds)** : la même mesure planche à plat, refaite à chaque déplacement des trépieds.
   - (c) **Repli** : axe hanche → cheville (comportement actuel du monde). La source est alors marquée `gravity_source = "body"` : les fautes sensibles à l'inclinaison (F2, F3, F5) ne sont cuées qu'à partir de « modéré ».
   - Option matérielle à discuter avec Ambaka : un accéléromètre à ~1 $ dans le boîtier donne la gravité en permanence.
2. **Latéral** = axe de la barre au repos, projeté sur le sol (ou ligne des hanches si pas de barre).
3. **Avant** = produit vectoriel, orienté vers la pointe des pieds (talon → pointe).
4. **Sol** = plan perpendiculaire à la gravité, passant sous les ancres de pied plantées (§6.5).
5. Conversions uniques, nommées par leur sens : `height_above_floor_m(point)`, `forward_of_m(point, reference)`, `lateral_of_m(point, reference)`, `sagittal_angle_deg(...)` (signé, positif vers l'avant). Interdiction de soustraire Y ou Z à la main ailleurs dans le package.
6. Tests obligatoires :
   - signe : hanche debout > 0,7 m au-dessus du sol ; barre qui monte pendant le tirage ; barre posée devant le pied ⇒ « avant » positif ;
   - invariance : monde synthétique incliné de 3° et 5° avec la gravité fournie ⇒ métriques identiques à ±2 mm et ±0,5° ;
   - repli : avec la gravité « body » sur un monde incliné, l'erreur mesurée est rapportée et les seuils effectivement élargis.

Observation pour Ambaka (hors périmètre, on n'y touche pas) : le même biais d'axe affecte probablement des métriques squat (inclinaison du tronc).

## 3. Isolation du squat (contrainte dure)

### 3.1 Architecture

```
main.py (~l. 452) : exercice = deadlift conventionnel ET NOWVA_ENABLE_DEADLIFT ?
   ├── non → pipeline_process.py (squat, inchangé, 0 ligne modifiée)
   └── oui → src/biomechanics/deadlift/process.py (nouveau sous-processus)
```

Le package `src/biomechanics/deadlift/` est construit **par composition**, sans hériter de `BiomechanicsPipeline` (un héritage chargerait le profil placeholder via `get_profile`, le moteur de règles et le compteur squat) :

| Module | Rôle | Réutilise (import, lecture seule) |
|---|---|---|
| `process.py` | Boucle de session : calibration caméra, apprentissage, séries, repos, IPC | `CameraCalibrationSession` ; helpers de `pipeline_process.py` importés s'ils sont au niveau module, sinon recopiés avec un commentaire d'origine |
| `camera_provider.py` | Sous-classe de `MultiCameraPoseProvider` qui **ajoute** une méthode `get_frames_and_pose()` renvoyant les 3 images synchronisées, les vues 2D et le squelette 3D. `get_pose()` n'est pas redéfinie, donc `CameraCalibrationSession` se comporte pareil | `MultiCameraPoseProvider`, `RTMPoseEstimator`, `DLTTriangulator` |
| `pipeline.py` | Par frame : chaîne pré-IK → IK → repère lifter → barre 3D → machine à états → métriques → règles | `build_preik_chain`, `AnalyticalIK`, `SegmentLengthEstimator` |
| `frame.py` | Repère du lifter (§2.2) | fichiers de calibration (lecture seule) |
| `bar_detector.py`, `bar_tracker_3d.py` | Détection multi-vues et suivi 3D de la barre (§6) | `calibration.undistort_keypoints`, Ultralytics |
| `rep_counter.py`, `metrics.py`, `rules/` | Phases, métriques, fautes F1–F10 | — |
| `setup_model.py` | Placement attendu (§5.4) | — |
| `ipc.py` | Messages deadlift (§9.2), choix des clés de cue | client IPC existant |
| `diagnosis/` | Graphe statique deadlift (§7) | types `DiagnosisResult` (lecture seule) |
| `voice/` | Textes des agents deadlift (apprentissage, calibration, récap) | — |
| `config/deadlift.yaml` | Seuils et paramètres deadlift | `config/biomechanics.yaml` reste intact |

Le BiLSTM (entraîné sur squats, `pipeline.py:243-259, 815-854`) n'est jamais instancié côté deadlift. Le profil placeholder `profiles/deadlift.py` reste en place mais n'est jamais chargé quand le drapeau est actif. On accepte la duplication (boucle de session, ≈ 150 lignes de calcul bayésien) ; une factorisation future ne se fera qu'avec l'accord d'Ambaka et sous golden master.

### 3.2 Drapeau

`NOWVA_ENABLE_DEADLIFT` (défaut `false`). Drapeau éteint ⇒ comportement de tous les fichiers partagés strictement identique à aujourd'hui, y compris les prompts et les cas d'erreur (aujourd'hui « sumo deadlift » lance le pipeline squat avec le profil placeholder : c'est conservé tel quel quand le drapeau est éteint).

### 3.3 Fichiers partagés modifiés : liste complète

Établie en suivant chaque message et chaque appel entre le pipeline, `main.py`, la voix, la base et l'écran. Tous les changements sont des **ajouts**, derrière le drapeau ou déclenchés uniquement par des messages que le squat n'émet jamais (champ `exercise` ou types `dl_*`).

| Fichier | Changement |
|---|---|
| `src/main.py` ~452 | Choix du script de sous-processus |
| `src/main.py` ~989-1035 | Branche d'affichage pour les messages portant `exercise = "conventional_deadlift"` ; ajout de `dl_setup_cue` à la liste de transfert vers la voix (~1035) |
| `src/visual/display.html` ~963, 1223, 1317, 1422 | Tuiles et libellés deadlift (trajectoire de barre, lockout…) quand le message porte `exercise` ; tuiles squat inchangées |
| `src/agent/agents/prompts/main_menu_prompt.py:26` | Texte « squats et deadlift conventionnel » si drapeau |
| `src/agent/agents/main_menu_agent.py` | `start_quick_exercise` : refus codé des exercices non pris en charge (sumo, RDL…) si drapeau ; recherche de calibration deadlift ; texte de progression (~640) selon l'exercice |
| `src/agent/agents/shared/helpers.py:94-99` | Alias ajoutés (« conventional deadlift », « Barbell Conventional Deadlift ») |
| `src/agent/agents/quickExerciseAgent.py:211` | Passe `exercise=` au `TeachingAgent` |
| `src/agent/agents/teaching_agent.py`, `calibration_agent.py`, `workout_agent.py` | Retour anticipé vers `deadlift/voice/` si exercice = deadlift ; code squat ni modifié ni réindenté |
| `src/agent/services/coaching_service.py` | Gestionnaires `fault` / `rep_complete` / `set_complete` / `diagnosis_complete` (~494-522 et suivants) : branche si `exercise` présent ; nouveau gestionnaire `dl_setup_cue` ; `BiomechanicsRecorder(exercise=...)` (~129, déjà supporté par `db/biomechanics_persistence.py:290`) ; requêtes de progression filtrées par exercice (~159-160) ; prompts de récap deadlift |
| `src/agent/services/coaching_orchestrator.py` | Textes de récap deadlift (~1241-1251, 1407, 1510) ; lecture d'un cue de placement via le chemin audio en cache existant |
| `src/agent/services/coaching_constants.py` | Clés de cue `dl_*` ajoutées (`CUE_TEXT_MAP`, `CUE_DISPLAY_LABELS`) |
| `src/agent/services/progress_context.py` | Libellés et filtre par exercice |
| `src/db/biomechanics_persistence.py:34` | Types `dl_*` ajoutés à `END_OF_REP_FAULT_TYPES` |
| `scripts/tools/generate_cue_audio.py` | Prompts audio `dl_*` ajoutés |

**Jamais modifiés** : `pipeline.py`, `pipeline_process.py`, `profiles/*`, `diagnosis/*`, `faults/*`, `coaching/{cue_cache,ipc_bridge,session_tracker}.py`, `calibration.py`, `utils/foot_contact.py`, `pose/multi_camera.py`, `barbell_tracking/*`, `config/biomechanics.yaml`.

**Collisions évitées par préfixe** : chaque type de faute et chaque clé de cue deadlift commence par `dl_`. Sinon il y aurait des collisions avec `PREEMPTIVE_TEXT["chest_up"]`, avec `END_OF_REP_FAULT_TYPES` (qui contient déjà `lockout`) et avec les requêtes non filtrées par exercice (`get_fault_progress`, `get_cue_effectiveness`).

### 3.4 Filet de sécurité squat (Jalon 0, avant toute ligne deadlift)

1. **Golden master pipeline squat** — `tests/test_biomechanics/test_squat_golden.py` :
   - Scénarios rejoués via le harnais multi-caméra de `tests/test_biomechanics/test_pipeline.py:329-392` (`_FakeProvider`, `_FakeClock`, `world_squat_points`, `squat_depth_profile`) : série propre, valgus, inclinaison, rep trop haute (`go_deeper`), perte de détection, asymétrie. Mono-caméra via `estimate_both` simulé ; BiLSTM via un faux déterministe.
   - Par frame : `model_dump(mode="json")` sans `latency_ms`, NaN → null. Puis `SessionTracker` + `IPCBridge` (client simulé) et `force_end_set` : on capture tous les messages IPC, y compris `diagnosis_complete`.
   - Déterminisme : `time.time` remplacé dans `session_tracker` et `ipc_bridge` en plus de l'horloge du pipeline ; `random` initialisé.
   - Comparaison : sérialisation canonique (flottants arrondis à 1e-9) dans l'environnement figé de `requirements.lock`. Les fixtures sont régénérées uniquement via une variable d'environnement explicite.
   - Lancé drapeau éteint **et** allumé.
2. **Golden master voix squat** — `tests/test_agent/test_squat_voice_golden.py` : le flux de messages IPC capturé ci-dessus est rejoué dans `CoachingService` + `CoachingOrchestrator` avec TTS, LLM et base simulés. On capture la séquence des cues joués, le texte exact des prompts de récap envoyés au LLM et les lignes écrites en base (`record_fault`, `record_rep`, `CueEvent`). Drapeau éteint et allumé. C'est ce filet qui couvre les fichiers voix du §3.3.
3. **Golden master des prompts** : texte des agents squat (menu drapeau éteint, apprentissage, calibration, séance).
4. **Test des alias** : tous les alias squat se résolvent en `SquatProfile` ; avec le drapeau, « sumo deadlift » est refusé et ne lance aucun pipeline.
5. **Manifeste de gel** : `tests/test_squat_freeze.py` vérifie l'empreinte SHA-256 des fichiers de la liste « Jamais modifiés » ; le mettre à jour exige l'accord d'Ambaka.
6. **Rejeu de vraies séances squat** : impossible aujourd'hui (`user_test_runs/` ne contient que des sorties, pas les vidéos brutes des 3 caméras). Dès que l'enregistreur existe (J1), on enregistre 3 à 5 séances squat brutes et on les rejoue dans le pipeline squat inchangé, via un fournisseur de rejeu injecté dans le test comme `_FakeProvider`. Snapshot des sorties.
7. **CI** : aucune aujourd'hui. Proposition : workflow GitHub Actions qui installe `requirements.lock` et lance `PYTHONPATH=src pytest tests/ -x` sur chaque PR (à valider avec Ambaka).
8. **Golden master deadlift** (à J3) : protège aussi le deadlift contre de futurs changements côté squat dans le code partagé (pré-IK, IK, triangulation).

Preuve que le filet fonctionne : modifier volontairement un seuil squat ou un texte de cue squat doit faire échouer le golden master correspondant.

## 4. Le mouvement : phases et machine à états

### 4.1 Phases

`APPROCHE` (debout, loin de la barre ou au-dessus) → `POSITION` (debout au-dessus de la barre, pieds plantés) → `PLACEMENT` (penché, mains sur la barre, immobile) → `TIRAGE` (décollage → passage des genoux → fin du tirage) → `LOCKOUT` → `DESCENTE` → `AU SOL`.

Deux moments d'arrêt servent à corriger : `POSITION` (pieds) et `PLACEMENT` (hanches, épaules). Pendant le tirage (≈ 1 s), aucun cue correctif.

### 4.2 Références personnelles prises à l'approche

Pendant `APPROCHE` debout immobile (≥ 1 s), on enregistre : angles de hanche, genou, tronc et coude debout ; hauteur des poignets bras pendants (= **hauteur de barre attendue au lockout**, moins le décalage poignet → prise) ; ancres des pieds (milieu du pied, §6.5). Les fautes F4, F5 et F7 sont mesurées **par rapport à ces références**, ce qui retire les biais de keypoints propres à chaque personne.

### 4.3 Signal de rep

1. Hauteur du centre de barre au-dessus du sol (barre 3D, §6).
2. Repli : milieu des poignets au-dessus du sol, moins le décalage poignet → barre appris quand les deux sont visibles. Le sol vient des pieds, pas de la barre, donc pas de circularité.
3. Jamais l'inversion du signal de hanche du placeholder actuel.

### 4.4 Machine à états (`deadlift/rep_counter.py`)

| Transition | Condition (valeurs initiales, `config/deadlift.yaml`) |
|---|---|
| `APPROCHE → POSITION` | debout, milieu du pied à moins de 15 cm de la barre (horizontalement), pieds immobiles ≥ 0,5 s |
| `POSITION → PLACEMENT` | mains à moins de 10 cm au-dessus de la barre, immobiles ≥ 0,3 s |
| `PLACEMENT → POSITION` / `APPROCHE` | se relève sans soulever (re-placement) |
| `PLACEMENT → TIRAGE` | barre > repos + 3 cm et vitesse > 0,10 m/s |
| `TIRAGE → LOCKOUT` | **posture** : hanche et genou à ≤ 10° de leur référence debout **et** barre ≥ hauteur de lockout attendue − 5 cm, maintenus ≥ 3 frames |
| `TIRAGE` immobile sous le lockout | rep « en lutte » : pas de lockout tant que la posture n'est pas atteinte |
| `TIRAGE → AU SOL` sans lockout | **rep ratée** : événement, non comptée |
| `LOCKOUT → DESCENTE` | vitesse < −0,10 m/s |
| `DESCENTE → AU SOL` | barre ≤ repos + 3 cm → **rep comptée et analysée** |
| `DESCENTE → TIRAGE` | touch-and-go : la vitesse redevient positive à moins de 5 cm du sol |
| `AU SOL → APPROCHE` | se relève sans la barre (fin de série possible) |

- **Comptage au sol** (un seul message `rep_complete`, §9.2). Simple et naturel en arrêt complet ; en touch-and-go, le compte tombe au contact.
- Barre lâchée depuis le lockout : rep comptée, **pas de faute** (normal avec des bumpers) ; simple information dans le récap.
- Petits ajustements de la barre au sol (< 3 cm) : ignorés.
- Fin de série : `AU SOL`/`APPROCHE` pendant `set_timeout_seconds` (30 s), ou « j'ai fini ».
- Hauteur de repos de la barre : médiane pendant `POSITION`/`PLACEMENT`, par série (s'adapte au diamètre des disques).
- Touch-and-go : aucun cue entre les reps (pas le temps) ; corrections au repos.

## 5. Fautes, mesures, seuils

Mesures dans le repère du lifter (§2.2). Seuils initiaux dans `config/deadlift.yaml`, **fixés définitivement sur données réelles (§8)**. Trois niveaux : léger / modéré / sévère. Le niveau minimal cué dépend du budget d'erreur (§5.6).

### 5.1 Fautes v1

| # | Faute | Phase de mesure | Mesure | Seuils initiaux | Cue |
|---|---|---|---|---|---|
| F1 | **Barre pas au-dessus du milieu du pied** | `POSITION` (debout, pieds plantés), recontrôle en `PLACEMENT` (la barre roule) | `bar_forward_of_midfoot_cm` | 3 / 5 / 8 cm | boucle fermée : « avance un peu / beaucoup » → « parfait » |
| F8 | **Hanches trop basses / trop hautes** au départ | `PLACEMENT` | Hauteur de hanche vs bande du modèle (§5.4) | hors bande de 4 / 7 / 10 cm | « Monte / descends un peu les hanches » |
| F9 | **Épaules derrière la barre** au départ | `PLACEMENT` | Épaule (articulation) vs barre, axe avant | derrière de 2 / 4 / 6 cm | « Épaules au-dessus de la barre » |
| F2 | **Hanches qui montent avant les épaules** | Décollage → passage des genoux | Variation de l'angle du tronc (signé) **moins** la variation prédite par le modèle de placement (≈ 0–5°) ; contrôle croisé : rapport des montées hanche / épaule | 10° / 15° / 20° ; rapport > 1,4 | « Poitrine et hanches ensemble — pousse le sol » |
| F3 | **Barre qui s'éloigne du corps** | Tirage | Écart avant max du centre de barre vs sa position au décollage | 3 / 5 / 8 cm | « Barre collée aux jambes » |
| F4 | **Lockout incomplet** | Point haut | Déficit d'extension de hanche ou de genou vs **référence debout** | 8° / 12° / 20° | « Serre les fessiers, finis debout » |
| F5 | **Hyperextension au lockout** | Point haut | Tronc en arrière vs **référence debout** (angle sagittal signé) | 8° / 12° / 18° | « Grandis-toi, ne te penche pas en arrière » |
| F6 | **Asymétrie** | Tirage | Décalage latéral du bassin vs milieu des pieds ; inclinaison de barre (différence de hauteur des extrémités) | 3 / 5 / 7 cm ; 3 / 5 / 7 cm | « Pousse pareil dans les deux pieds » |
| F7 | **Bras pliés** | Tirage | Flexion du coude vs **référence debout** | 15° / 25° / 35° | « Bras longs et tendus » |

La rotation du tronc est retirée de F6 en v1 : une prise mixte (main en pronation + main en supination) la déclenche à tort. Le type de prise est demandé au lifter (« prise mixte ? ») et enregistré.

### 5.2 Fautes v1.1

Arrachage sans mise en tension (pic d'accélération au décollage) ; descente non contrôlée ; genoux qui avancent avant le passage de la barre à la descente ; « hitching » (genou qui se replie au lockout) ; position de la tête ; largeur de prise et de pieds ; genoux qui rentrent ; talons qui décollent ; perte de vitesse de barre (fatigue, base du diagnostic « charge trop lourde » et du VBT).

### 5.3 Dos rond : proxys honnêtes

- Les keypoints (COCO 17 + pointes + talons = 21) n'ont aucun point sur la colonne : on **ne mesure pas** le dos rond et on ne l'affirme jamais.
- On mesure une « perte de position » : F2 + F3 + F9.
- Signal expérimental : raccourcissement de la distance épaule ↔ hanche vs la longueur debout. Le bruit par frame d'un keypoint triangulé est de **1,6 à 2,4 cm** (`utils/segment_lengths.py:19-21`), comparable à l'effet attendu (1 à 3 cm). Le signal n'est donc testé qu'en moyenne sur la phase de tirage. Il n'est activé que si les données annotées montrent une séparation nette entre reps « dos rond » et reps propres (AUC ≥ 0,8 sur lifters jamais vus).
- Cues comportementaux uniquement (« dos plat, gaine-toi »), jamais médicaux ni de « sécurité ».
- La `BackRoundingRule` actuelle n'est pas utilisée : elle mesure la variation d'inclinaison du tronc, qui est le mouvement normal du deadlift.

### 5.4 Modèle du placement attendu (cœur de l'IP)

Équivalent deadlift de `expected_trunk_lean_geometric` du squat : la bonne position de départ pour **ce corps-là**.

- **Contraintes**, sous forme de bandes et non de points :
  - barre au-dessus du milieu du pied ;
  - articulation de l'épaule 0 à 6 cm devant la barre (les omoplates au-dessus de la barre placent l'articulation un peu devant) ;
  - bras verticaux ;
  - tibia en contact avec la barre.
- **Entrées** :
  - longueurs tibia, fémur, torse, pied (`SegmentLengthEstimator` mesure tout sauf les bras) ;
  - **longueur des bras** : mesurée par un estimateur propre au deadlift (épaule → coude → poignet), le fichier partagé n'est pas modifié ;
  - décalage poignet → prise : 6–9 cm, la barre est dans la main sous l'articulation du poignet ; appris par personne quand barre et poignets sont visibles ensemble ;
  - distance axe de barre ↔ ligne cheville–genou au contact : rayon de barre (1,4 cm) + épaisseur de tissu devant le tibia (3–5 cm), valeur initiale 5 cm ;
  - hauteur de repos de la barre, mesurée.
- **Résolution** : l'épaule est placée dans sa bande au-dessus de la barre (hauteur = barre + prise + bras) et la cheville est fixée. La chaîne cheville → genou → hanche → épaule n'a alors qu'un degré de liberté. La contrainte de contact du tibia admet **deux solutions** : on garde celle avec le genou devant la cheville et une flexion du genou dans [40°, 120°]. S'il n'y a aucune solution, la morphologie est hors modèle : on désactive F8 et F9 pour cette personne et on le journalise.
- **Sorties** : bande de hauteur de hanche, angle du tronc attendu, flexion du genou attendue.
- **Utilisation** : F8, F9, correction de F2, diagnostic, fantôme optionnel du placement idéal.
- **Validation** :
  - tests sur géométrie connue ;
  - cohérence (fémurs longs ou bras courts ⇒ dos plus horizontal) ;
  - comparaison avec les placements annotés « bons » par le coach : erreur médiane de hauteur de hanche ≤ 4 cm.

### 5.5 Personnalisation

- F4, F5, F7 : relatives à la référence debout de la séance (§4.2).
- F1, F3, F6 : distances absolues (l'anatomie ne change pas la cible).
- F8, F9 et F2 : relatives au modèle personnalisé §5.4.
- Une série d'échauffement mesure le bruit de chaque métrique pour cette personne : seuil = max(seuil de base, 3 × bruit), plafonné par une limite qu'aucune calibration ne dépasse. On n'utilise jamais le « pic observé + marge » du squat, qui peut normaliser une faute.
- Stockage sous `movement_pattern = "conventional_deadlift"`. Le chemin deadlift ne passe jamais par le repli `or "squat"` de `pipeline_process.py:1073`.
- Les mesures corporelles déjà stockées pour le squat sont lues (lecture seule) comme point de départ, puis remesurées dans la séance.

### 5.6 Budget d'erreur

Règle : on ne cue un niveau que si l'erreur de mesure (95e percentile, mesurée contre la vérité terrain §8.4) est ≤ 1/3 du seuil de ce niveau. Sinon, on ne cue qu'à partir du niveau supérieur.

| Faute | Seuil léger | Erreur max pour cuer « léger » | Erreur attendue (hypothèse à vérifier) | Niveau cué au départ |
|---|---|---|---|---|
| F1 | 3 cm | 1 cm | ≤ 1 cm (barre statique + ancres de pied moyennées sur ≥ 15 frames) | léger |
| F3 | 3 cm | 1 cm | 1–2 cm (dynamique) | modéré tant que non prouvé |
| F6 latéral | 3 cm | 1 cm | 1–2 cm | modéré |
| F2 | 10° | 3,3° | 2–4° | modéré tant que non prouvé |
| F4 / F5 | 8° | 2,7° | 2–3° (relatif à la référence) | léger si prouvé, sinon modéré |
| F7 | 15° | 5° | 3–5° | léger |
| F8 | 4 cm | 1,3 cm | 2 cm + erreur du modèle | modéré |
| F9 | 2 cm | 0,7 cm | 1–1,5 cm | modéré |

## 6. Barre au sol : détection et suivi 3D

### 6.1 Existant

- `BarbellDetector` (`barbell_tracking/detector.py`) : YOLO11n-pose, renvoie uniquement la meilleure détection (`:113`).
- Les poids `models/barbell_keypoints.pt` ne sont pas dans le repo ; on ne connaît ni leurs données d'entraînement ni leur origine.
- Le suivi est désactivé par défaut (`config.py:145`) ; en multi-caméra, il ne tourne que sur la caméra principale, en 2D (`pipeline.py:613-624` ; `multi_camera.py:307-357` ne renvoie que l'image principale).
- Les extrémités de la barre ne sont triangulées que dans la calibration caméra, pour l'échelle, puis jetées (`person_calibration.py:599-601, 852, 968`).
- `BarPathRule` mélange l'axe latéral (avec détection) et l'axe avant/arrière (repli poignets) : non réutilisée.

### 6.2 Keypoints

Pour une barre chargée, 2 keypoints = **centre de la face extérieure du disque extérieur**, à gauche et à droite (sur l'axe de la barre). Un disque de 45 cm est grand, contrasté, visible de face comme à 45°. On confirme sur les premières images réelles (J1) avant d'annoter en masse. Barre vide (sans disques) : extrémités du manchon. Les deux définitions sont annotées comme deux classes distinctes.

### 6.3 Données et entraînement

1. Récupérer les poids actuels chez Ambaka et les évaluer tels quels (J1). Décision : affiner ce modèle ou repartir de YOLO11n-pose.
2. Images extraites des enregistrements bruts (§8.3) : barre au sol, en tirage, au lockout, barre rangée dans les J-hooks (distracteur), barre vide / bumpers / fonte, couleurs, éclairages, tenues, lifters. Les 3 caméras.
3. Annotation dans **CVAT auto-hébergé** (le même outil sert aux étiquettes de reps, §8.3). 2 000 à 3 000 images, équilibrées par caméra et par phase, avec des négatifs.
4. Entraînement Ultralytics (déjà en dépendance), export ONNX puis TensorRT FP16.
5. Évaluation par lifter : rappel ≥ 95 % sur barre au sol ; erreur keypoint (px) ; faux positifs sur barre rangée ≤ 1 %.

### 6.4 Exécution en série

1. `camera_provider.py` fournit les 3 images synchronisées (§3.1).
2. `bar_detector.py` : appel direct du modèle Ultralytics en **lot sur les 3 vues**, en renvoyant **toutes** les détections au-dessus du seuil (pas seulement la meilleure). Nouvelle classe, `BarbellDetector` n'est pas modifié.
3. **Correction de distorsion** des keypoints (`calibration.undistort_keypoints`) avant triangulation.
4. Association multi-vues : on garde la combinaison dont la triangulation a la plus petite erreur de reprojection, compatible avec les mains et les pieds du lifter. Cela élimine la barre rangée.
5. Gauche/droite étiquetés par la ligne des épaules (même principe que `_label_bar_ends_by_shoulders`, recopié).
6. DLT sur les 2 extrémités (≈ 40 lignes dans le package ; les fonctions de `triangulator.py` sont privées). Rejet si l'erreur de reprojection dépasse le seuil.
7. Kalman 3D à vitesse constante par extrémité, prédiction pendant les trous courts.
8. Sortie `BarState3D` dans le repère du lifter : centre, hauteur au-dessus du sol, avant/arrière vs milieu du pied, inclinaison (cm et °), vitesse (m/s), `source` (`bar` / `wrist_proxy`), confiance.

### 6.5 Sol et milieu du pied

`foot_contact.py` ne publie pas ses ancres et reste intouché. Le package deadlift :
- verrouille les ancres de pied (talons, pointes) dans le repère du lifter pendant `APPROCHE`/`POSITION` (pieds immobiles, moyenne sur ≥ 15 frames) ;
- milieu du pied = milieu talon ↔ pointe, **verrouillé avant le tirage**, donc insensible aux occultations par les disques pendant le tirage ;
- sol = plan perpendiculaire à la gravité passant sous les ancres, moins le décalage keypoint → sol (talon ≈ 3–5 cm, pointe ≈ 1–2 cm), mesuré une fois sur le rig avec la planche à plat ;
- contrôle croisé : barre au repos = rayon du disque au-dessus du sol ;
- remise à zéro quand la calibration caméra change.

### 6.6 Repli sans barre

Milieu des poignets moins le décalage appris. F1, F3 et l'inclinaison de F6 passent en `source = wrist_proxy` : seuils élargis et cue au niveau « modéré » minimum. Si la confiance est trop basse, la faute n'est pas émise.

### 6.7 Coût edge

YOLO11n-pose × 3 vues est le principal coût ajouté. Il est **mesuré sur Jetson dès J1** (§10). Leviers dans l'ordre :
1. TensorRT FP16 ;
2. entrée 480 px ;
3. détection 1 frame sur 2 avec prédiction Kalman entre les deux ;
4. recadrage autour des mains et des pieds ;
5. seulement les 2 caméras à 45°.

## 7. Diagnostic après la série (graphe statique)

### 7.1 Principe

Même méthode que le squat : symptômes → causes candidates → probabilité de départ écrite à la main × test de preuve → normalisation avec fuite → noisy-OR. Sortie au **même schéma `DiagnosisResult`**.

Le module est séparé, `deadlift/diagnosis/` : `symptoms.yaml`, `causes.yaml`, `evidence_tests.py`, `parameter_deltas.py`, `engine.py`. Le moteur squat charge ses graphes en globales à l'import (`diagnosis/graph/loader.py:114-116`), il n'est donc pas réutilisable sans modification.

Sorties simples pour un petit LLM local : par hypothèse, `cause_id`, niveau, score, **un** delta chiffré, une phrase.

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
| `dl_uneven_stance_or_grip` | 1 | « décale ta prise de X cm » (poignets vs centre de barre) | F6 |
| `dl_weight_too_heavy` | 2 | « baisse d'environ 10 % » | dégradation au fil de la série |
| `dl_weak_off_floor` | 3 | — | F2 malgré un bon placement |
| `dl_hip_hamstring_mobility` | 3 | — | F8 « trop bas impossible à corriger » répété |
| `dl_unilateral_weakness` | 3 | — | F6 persistant |
| `dl_anthropometric_context` | 0 | — | « fémurs longs / bras courts : dos plus horizontal attendu » |

### 7.4 Score de rep

5 dimensions dans [0, 1], poids statiques :
- placement (F1, F8, F9) : 25 %
- coordination (F2) : 25 %
- trajectoire de barre (F3) : 20 %
- lockout (F4, F5) : 15 %
- symétrie (F6) : 15 %

Pas de « profondeur ».

### 7.5 Fantôme

Le correcteur de keypoints du squat n'est pas réutilisé. En option pour la démo : le placement idéal du §5.4 affiché comme fantôme statique.

## 8. Données

### 8.1 Connaissance statique

`docs/deadlift/KNOWLEDGE.md` contient :
- les phases ;
- la définition exacte de chaque métrique (repère, signe, phase) ;
- les seuils initiaux et leur justification ;
- les textes des cues ;
- le modèle géométrique.

Sources : modèle de placement classique du coaching (barre au-dessus du milieu du pied, omoplates au-dessus de la barre, bras verticaux), manuels de référence en préparation physique. Revue par Ambaka et idéalement par un coach de force externe. Chaque seuil porte « à valider sur données ».

### 8.2 Enregistreur brut et rejeu (J1, prérequis de tout le reste)

- `scripts/tools/record_rig.py` : outil autonome qui ouvre les 3 caméras via `multi_capture` (import) et écrit :
  - une vidéo par caméra (H.264 haute qualité ou MJPEG) ;
  - l'horodatage de chaque frame ;
  - une copie des fichiers de calibration ;
  - les métadonnées (lifter, charge, disques, prise, chaussures).

  Il ne passe par aucun pipeline.
- `deadlift/replay_provider.py` : relit un enregistrement en exposant la même interface que le fournisseur caméra. Il sert au pipeline deadlift hors ligne et, injecté dans un test, au rejeu des séances squat dans le pipeline squat inchangé (§3.4.6).
- Volume : ≈ 5–10 Mo/s pour 3 × 720p30, soit ≈ 5 Go pour 10 min. Stockage local, sauvegarde chiffrée.

### 8.3 Données réelles

**Round 1** (équipe, semaines 1–2), 2–3 personnes :
- barre vide et charges légères ;
- reps propres et fautes volontaires sans danger (F1, F2, F3, F4, F5, F6, F7, F8, F9) ;
- le dos rond est montré uniquement avec barre vide ou bâton.

**Round 2** (pilote), 10–15 lifters de 1,55 à 1,95 m, aux ratios fémur/torse/bras variés, débutants à confirmés :
- séries naturelles à RPE ≤ 8 ;
- et **séries de fautes scénarisées** pour garantir des exemples positifs.

**Jeu de test** : ≥ 5 lifters jamais vus à l'entraînement ni au réglage des seuils, avec ≥ 30 exemples positifs par faute v1 (fautes scénarisées comprises). Chaque métrique est rapportée avec son intervalle de Wilson à 95 %.

**Annotation** (CVAT, vidéo des 3 vues) :
- par rep : chaque faute (présente / niveau) ;
- horodatage des événements (décollage, passage des genoux, lockout, sol) sur un sous-ensemble, pour évaluer la machine à états seule ;
- 2 annotateurs sur 20 % des reps ; accord cible κ ≥ 0,6 par faute ;
- désaccords tranchés par Ambaka ;
- toute faute avec κ < 0,6 est redéfinie avant d'être cuée.

**Volumes v1** : ≥ 300 reps annotées, ≥ 50 positifs par faute (entraînement et réglage), plus le jeu de test ci-dessus ; 2 000–3 000 images annotées pour la barre.

**Métadonnées** : charge, diamètre des disques, type de prise (double / mixte / crochet), chaussures, ceinture, taille et envergure ; sur un sous-ensemble, longueurs de segments mesurées au mètre ruban.

**Consentement** écrit, anonymisation, stockage local.

### 8.4 Vérité terrain

| Grandeur | Méthode |
|---|---|
| Position statique de la barre / F1 | Scotch au sol et gabarit de pied : barre posée à 0 / 3 / 6 / 10 cm devant le milieu du pied |
| Trajectoire dynamique de la barre (F3, vitesse) | **Marqueurs ArUco collés au moyeu des disques**, triangulés par les mêmes caméras calibrées (méthode indépendante du détecteur) ; en option, un capteur de position linéaire à câble pour la hauteur et la vitesse |
| Angles du tronc et de la hanche (F2, F4, F5, F8) | **Caméra de référence sagittale** (téléphone 60 fps, perpendiculaire) avec marqueurs adhésifs sur épaule, hanche, genou, cheville ; angles 2D comparés |
| Gravité | Planche ChArUco à plat (§2.2) ; vérification au niveau à bulle |
| Événements | Horodatages annotés (§8.3) |

### 8.5 Données synthétiques (simulateur)

Le simulateur à vérité terrain existe pour le squat dans `.claude/preik-audit/harness/preik_harness/`. Travaux :
- chemins en dur corrigés (`__init__.py:13`) ;
- `runner.py:404` rendu paramétrable par exercice ;
- générateur deadlift (≈ 200–300 lignes) : départ au sol, mains liées à la barre, `pose_from_params` réutilisé, barre au sol (`barbell.py` la met aujourd'hui sur le dos), occultation par les disques, **monde incliné** pour tester §2.2 ;
- scénarios : propre, F1–F9, touch-and-go, barre lâchée, rep ratée, re-placement, plusieurs morphologies.

Usage : tests unitaires et d'intégration, robustesse au bruit, invariance à l'inclinaison. **Jamais pour fixer les seuils finaux.**

### 8.6 Questions auxquelles les données doivent répondre avant la démo

1. Qualité de la pose en position penchée (visage vers le sol, bras devant les genoux) : keypoints perdus par caméra et par phase.
2. Occultation des pieds et des tibias par la barre et les disques, par caméra.
3. Erreur réelle de chaque métrique (§5.6) ⇒ niveau minimal cué.
4. Bruit sur reps propres ⇒ seuils.
5. Précision et rappel par faute ⇒ cibles du §1.
6. Signal expérimental « raccourcissement épaule ↔ hanche » : on le garde ou on l'abandonne.

## 9. Coaching vocal et messages

### 9.1 Parcours utilisateur

- **Entrée** : session « exercice rapide » (« on fait du deadlift »), et séance programmée **dont le premier exercice est le deadlift**. Aujourd'hui le sous-processus de pose est lancé une fois par séance avec le premier exercice (`main.py:1065-1099`) : passer de squat à deadlift au milieu d'une séance demande un redémarrage du sous-processus, prévu en v1.1 et à valider avec Ambaka car cela touche le parcours squat.
- **Refus** : sumo, RDL et toute variante non prise en charge sont refusés par le code (`start_quick_exercise`), avec une réponse claire.
- **Apprentissage deadlift** (première séance), module de textes séparé :
  1. pieds largeur de hanches, barre au-dessus du milieu du pied ;
  2. prise juste à l'extérieur des jambes ;
  3. tibias à la barre ;
  4. épaules au-dessus de la barre, dos plat ;
  5. mise en tension, puis pousser le sol.

  Ensuite, 2–3 reps à la barre vide qui servent aussi à mesurer le bruit (§5.5).
- **Calibration caméra** : si aucun fichier de rig n'existe, la calibration actuelle demande 2 squats lents au poids du corps. On la garde (elle calibre les caméras, pas l'exercice) et l'agent deadlift l'annonce.
- **Guidage du placement en boucle fermée** (moment fort de la démo), dans le pipeline deadlift :
  - pendant `POSITION`, puis pendant `PLACEMENT`, on compare F1, F8 et F9 aux cibles ;
  - si l'écart est hors tolérance, on envoie un cue directionnel (« avance un peu » / « avance encore », « hanches un peu plus haut ») toutes les 1,5 s au plus ;
  - une fois dans la tolérance, on envoie « parfait » ;
  - 4 cues maximum par placement.

  C'est le même principe que le moniteur d'ajustement squat (`coaching_orchestrator.py:381-520`), réimplémenté côté deadlift sans toucher à l'orchestrateur.
- **Politique de cues pendant la série** :
  - jamais de cue correctif pendant le tirage ;
  - 1 correction maximum par rep, envoyée au sol ;
  - priorité : placement (F1/F8/F9) > perte de position (F2/F3) > lockout (F4/F5) > asymétrie (F6) > bras (F7) ;
  - aucune correction entre deux reps en touch-and-go.
- **Audio** : cues directionnels à deux intensités (« un peu » / « encore ») plutôt qu'un nombre exact. Les chiffres (« 4 cm ») sont dits par le LLM dans le récap. Clips pré-générés hors ligne, lecture locale, repli TTS existant.
- **Récap de série** : diagnostic deadlift → LLM local, prompt deadlift séparé.

### 9.2 Contrat de messages pipeline deadlift → main.py → voix

Les messages deadlift portent tous `exercise = "conventional_deadlift"`. Les messages squat ne changent pas (absence du champ = squat).

| Type | Émis quand | Champs | Consommateurs (branche deadlift) |
|---|---|---|---|
| `cache_cues` | Début de séance | clés `dl_*` | `coaching_service` (cache audio existant) |
| `dl_setup_cue` (nouveau) | Guidage de placement | `cue_key`, `kind` (correction / confirmation), `metric`, `value_cm`, `target_cm` | `main.py` (transfert), `coaching_service` → lecture audio en cache |
| `fault` | Fin de rep (1 par rep max) | `fault_type` `dl_*`, `severity`, `cue_key` `dl_*`, `message`, `rep_number`, `value`, `source` | `coaching_service` → `orchestrator.on_fault` ; enregistreur en base |
| `rep_complete` | Barre au sol | `rep_number`, `set_number`, `is_clean`, `faults_in_rep`, `bar_rise_m`, `concentric_time_s`, `lockout_ok`, `setup` {F1/F8/F9}, `bar_source` | `coaching_service` (branche), enregistreur, affichage (`main.py` ~989) |
| `set_complete` | Fin de série | `total_reps`, `clean_reps`, `fault_summary`, `failed_reps` | affichage, récap |
| `diagnosis_complete` | Après la série | `DiagnosisResult` + `scoring.per_dimension` (5 dimensions deadlift) | `coaching_service`, affichage |
| `rest_complete`, `frame_data` | Comme le squat | `frame_data` avec `rep_phase` deadlift | inchangé |

Enregistrement : `BiomechanicsRecorder(exercise="conventional_deadlift")`. Les séances, reps et `CueEvent` sont étiquetés, ce qui prépare l'apprentissage futur sans le construire.

## 10. Edge (Jetson)

- **J1** : mesure sur Jetson Orin Nano Super de RTMPose 3 vues + YOLO11n-pose 3 vues (TensorRT FP16) + chaîne complète.
- Nouveaux calculs (machine à états, géométrie, modèle de placement, diagnostic) : solutions fermées, négligeables (cible < 1 ms).
- **Mode dégradé** défini si le budget est dépassé : pose à 30 Hz et barre à 15 Hz avec prédiction Kalman. Sinon, tout le pipeline à 15 Hz (acceptable pour le placement statique ; à vérifier pour le chronométrage du décollage, §1).
- Rien de nouveau dans la boucle conversationnelle ne dépend du cloud. Annotation et entraînement : hors ligne.

## 11. Jalons, critères d'acceptation, estimations

Estimations pour 1 ingénieur à temps plein, à affiner. Avec 2 personnes, J3/J4 et J2/J4 se parallélisent.

| Jalon | Contenu | Critère d'acceptation | Dépend de | Estim. |
|---|---|---|---|---|
| **J0 — Filet squat** | Environnement `requirements.lock`, golden masters pipeline et voix, prompts, alias, manifeste de gel, drapeau sans effet, CI (si validée) | Tous verts sur `main` ; une modif volontaire d'un seuil squat et d'un texte de cue squat fait échouer le bon test | — | 4–6 j |
| **J1 — Fondations** | Enregistreur brut + rejeu ; repère du lifter + gravité (planche à plat) + tests ; `KNOWLEDGE.md` revu ; modèle de placement + tests ; capture round 1 ; mesure Jetson | Revue signée ; tests de signe et d'inclinaison verts ; ≥ 1 h de capture brute ; keypoints de barre décidés ; chiffres Jetson | J0 | 2 sem. |
| **J2 — Simulateur deadlift** | Générateur, scénarios, monde incliné | Chaque scénario produit sa vérité terrain | J1 | 1 sem. |
| **J3 — Cœur deadlift (pose + repli poignets)** | Fournisseur caméra, machine à états, références debout, métriques, F1–F9, guidage de placement, golden master deadlift | Simulateur : 100 % des reps comptées, événements ≤ 100 ms, chaque faute injectée détectée, scénario propre sans faute ; rejeu round 1 sans crash | J2 | 2 sem. |
| **J4 — Barre 3D** | Annotation CVAT, entraînement, export, tracker 3D, association multi-vues, sol, vérité terrain ArUco | Rappel ≥ 95 % ; erreur 3D statique ≤ 1 cm, dynamique ≤ 1,5 cm vs ArUco ; temps Jetson dans le budget ou mode dégradé validé | J1 | 2–3 sem. |
| **J5 — Intégration** | `deadlift/process.py`, bifurcation, contrat §9.2, fichiers partagés §3.3, voix, affichage, base | Séance complète sur le rack (« deadlift » → placement guidé → reps comptées → cue → récap) ; golden masters squat pipeline et voix identiques, drapeau éteint comme allumé | J0, J3, J4 | 2 sem. |
| **J6 — Validation réelle** | Annotation round 1 → seuils et budget d'erreur ; round 2 pilote → jeu de test | Cibles du §1 sur lifters jamais vus, avec intervalles de Wilson ; κ ≥ 0,6 | J5 | 3 sem. |
| **J7 — Diagnostic deadlift** | Graphe statique, score, récap | Cause n°1 = annotation du coach dans ≥ 70 % des séries annotées | J6 | 1–1,5 sem. |
| **J8 — Démo** | Scénario héros : placement guidé (« avance un peu… parfait ») puis F2 corrigée à la rep suivante ; plan B : F1 seule | 10 démos consécutives sans erreur ; budget Jetson respecté | J7 | 1 sem. |

Total indicatif : 15–17 semaines pour 1 personne, ≈ 9–10 semaines pour 2. La capture réelle démarre dès J1, en parallèle du code.

## 12. Risques et parades

| Risque | Parade |
|---|---|
| Toucher au squat par accident | Sous-processus séparé, composition, drapeau, golden masters pipeline + voix, manifeste de gel, CI |
| Axe vertical faux | Gravité mesurée (§2.2), tests d'inclinaison, seuils élargis en repli |
| Dos rond non mesurable | Proxys, vocabulaire comportemental, signal expérimental soumis aux données |
| Pose dégradée en position penchée | Mesure au round 1 ; affinage RTMPose sur nos images si besoin (hors v1) |
| Occultation des pieds et tibias | Milieu du pied verrouillé avant le tirage ; mesure par caméra |
| Détecteur de barre qui ne généralise pas | Données variées, découpage par lifter, négatifs |
| Seuils inventés | Valeurs initiales seulement ; budget d'erreur ; données réelles avant la démo |
| Trop peu d'exemples de fautes rares | Fautes scénarisées dans le jeu de test, intervalles de Wilson |
| Annotations incohérentes | Double annotation, κ ≥ 0,6, arbitrage |
| Budget Jetson | Mesure à J1, mode dégradé défini |
| Duplication squat ↔ deadlift qui diverge | Golden master deadlift ; factorisation future seulement avec accord |
| Prise mixte qui fausse l'asymétrie | Rotation du tronc retirée de F6 ; prise enregistrée |
| Collisions de noms dans les données | Préfixe `dl_` partout |

## 13. Hors périmètre v1

Sumo, RDL et autres variantes ; changement d'exercice au milieu d'une séance ; apprentissage des probabilités ou du LLM ; fantôme animé ; ML (TCN) pour le dos rond ; VBT complet ; toute modification d'un fichier squat.

## 14. Questions restantes pour Ambaka

1. Qui annote (Ambaka, un coach externe) ?
2. Touch-and-go compté en v1 (proposé : oui, sans correction entre les reps) ?
3. Où sont les poids actuels du détecteur de barre et sur quelles images ont-ils été entraînés ?
4. OK pour la CI GitHub Actions ?
5. OK pour la duplication assumée (boucle de session, moteur de diagnostic) ?
6. Un accéléromètre dans le boîtier est-il envisageable (gravité permanente) ?
7. Quels disques dans la salle de démo (bumpers 45 cm ?) ?
