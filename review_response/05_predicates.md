# Task 5 — Contact vs. Non-Contact Predicates (Answers Reviewer A)

## 1. Predicate Partition Summary (`[LOG]`)

- **Source of $\mathcal{V}_{\text{contact}}$:** Hardcoded Python set `CONTACT_VERBS` in `vynix_fewshot_adapter.py:79-88` (`64` strings) `[LOG]`.
- **Confirmation of Non-Contact Behaviour:** In `compute_soft_geometric_gate` (`vynix_fewshot_adapter.py:175-188`), `if verb in CONTACT_VERBS:` is the sole branch condition; any verb **not** in `CONTACT_VERBS` unconditionally returns `1.0` (`gamma = 1.0`) `[LOG]`.
- **Schema Mismatch Note (`[LOG]`):** Cross-referencing `CONTACT_VERBS` (64 strings) against all 600 rows of `list_action.csv` (`117` unique HICO-DET verbs including `no_interaction`) reveals that **5 strings in `CONTACT_VERBS` do not appear as `vname` in `list_action.csv`**: `grab`, `mount`, `play_with`, `taste`, `touch`. Consequently, out of the **117** actual HICO-DET verbs, **59** are gated as contact verbs and **58** are treated as non-contact (`gamma = 1.0`). (See `review_response/05_predicates.csv`.)

---

## 2. Questionable Verb Assignments Flagged

| Verb | Current Setting | Flag / Concern |
|---|---|---|
| `fly` | Contact (`1`) | Ambiguous: *fly a kite* (`d_box > 0`, non-contact!) vs. *fly an airplane* (inside/contact). Gating `fly` as contact penalizes *fly a kite* when person and kite boxes do not overlap. |
| `hose` | Contact (`1`) | Distant interaction: *hose a car/dog/plant* uses a water stream, so human and target boxes frequently have `IoU == 0`. |
| `herd` | Non-contact (`0`) | Generally reasonable (herding sheep/cows from a distance), though close contact can occur. |
| `feed`, `scratch`, `stick`, `stir`, `blow`, `break`, `clean`, `cook`, `light`, `paint`, `repair`, `sign`, `tag`, `throw`, `toast`, `adjust`, `assemble`, `block`, `greet`, `set`, `serve` | Non-contact (`0`) | **Major under-coverage:** Many of these require physical contact or tool-mediated contact (e.g., `scratch` a dog/cat, `paint` a vase/hydrant, `repair` a bike/laptop, `cook` food, `adjust` a tie/skis, `greet` via handshake, `clean` a sink/oven, `break` a bat), yet they are excluded from `CONTACT_VERBS` and receive `gamma = 1.0`. |
| `watch`, `point`, `wave` | Non-contact (`0`) | Correctly assigned `0` (purely visual/gestural, no physical contact). |
| `ride`, `sit_on`, `hold`, `carry`, `eat` | Contact (`1`) | Correctly assigned `1` (physical support/grasp/ingestion). |

---

## 3. Ready-to-Paste 2-Sentence Description for the Paper

> We partition the 117 HICO-DET verb predicates into a manually curated set of 59 active physical-contact predicates (e.g., *hold, carry, ride, sit_on, eat, cut*, out of 64 entries in `CONTACT_VERBS`, 5 of which—*grab, mount, play_with, taste, touch*—do not appear as canonical HICO-DET verb labels) and 58 non-contact or unconstrained predicates (including visual/gestural verbs such as *watch, point, wave* and `no_interaction`). Non-contact predicates bypass spatial attenuation (`gamma = 1.0` unconditionally), whereas contact predicates are attenuated by the soft Gaussian distance gate whenever the human and object boxes are disjoint (`IoU = 0`).
