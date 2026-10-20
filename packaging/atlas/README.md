# msk-pipe attachment atlas `lhdl`

Muscle attachment areas (origin and insertion outlines) on the bones of one subject, used
by msk-pipe (`attachments` step, plugin `atlas_registration`) to place the attachment areas
on a new subject by registering the atlas bones onto the subject's bones.

**Licence:** CC BY-NC-SA 2.0 BE, derived from the LHDL dataset (see `LICENSE`). Not part
of the Apache-2.0 licence of msk-pipe.

## Contents

| Path | Content |
|---|---|
| `atlas.json` | muscle -> stem, origin/insertion file, bone and model body |
| `bones/*.obj` | atlas bones: `pelvis_with_sacrum`, `pelvis_no_sacrum`, `femur_<s>`, `tibia_<s>` (tibia + fibula), `patella_<s>` |
| `muscles/<muscle>/<Stem>_{Ori,Ins}.vtk` | closed attachment outlines, legacy VTK polydata (Muscle Wrapping 2.x layout) |

Coordinates: world RAS (as NIfTI), millimetres.

## Source

LHDL Cadaver #1 (female donor), Istituto Ortopedico Rizzoli and Universite Libre de
Bruxelles, project LHDL (EU FP6 IST-2004-026932), distributed through Physiome Space:

- bones: surfaces segmented from the lower-limb CT (`S032_*_B000_CTA_*`),
- attachment areas: landmark clouds digitised during dissection (`Ori_*`, `Ins_*`).

## Changes made to the original data

1. Converted from the MAF scene frame (LPS) to RAS; units kept in mm.
2. Bones: left and right iliac bones merged into `pelvis_no_sacrum`, plus the sacrum into
   `pelvis_with_sacrum`; fibula merged with the tibia.
3. Attachment areas: only muscles used by msk-pipe; several landmark clouds of one area
   merged into one closed outline (digitisation order kept, clouds joined by 2-opt);
   clouds lying on another bone than the muscle's attachment removed.
4. Areas missing in the dissection data (right gluteus minimus, right obturator internus
   and externus, some insertions of adductors, pectineus, piriformis, quadriceps and
   biceps femoris) taken from points placed manually on the same LHDL bones in the
   bachelor thesis of P. Mrnka (2026).
5. No mirroring between sides: areas missing on one side are absent.

## Citation

Cite the LHDL dataset (Viceconti et al.; www.livinghuman.org) and msk-pipe
(`CITATION.cff` in the repository).
