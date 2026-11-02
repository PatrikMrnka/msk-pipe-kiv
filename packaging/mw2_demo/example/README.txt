Example input: LHDL Cadaver #1, lower-limb CT
=============================================

06_mw2_input of a msk-pipe run on the lower-limb CT of LHDL Cadaver #1 (right side, default
settings); demo.json ("example") names the msk-pipe version and the run. Only the inputs of
Muscle Wrapping are included, no results.

  setup_MuscleGeneratorTool.xml   Muscle Wrapping setup (fibres per muscle, motion)
  bone_model.osim, Geometry\      skeletal model (pystaple / STAPLE) and bone surfaces
  hip_flexion_r.mot               hip flexion from the rest pose
  muscles\<muscle>\               muscle surface (.obj), origin and insertion (.vtk)
  export.json                     what msk-pipe checked, repaired and excluded

Changes made to the original data: bones and muscles segmented automatically from the
CT (TotalSegmentator, MuscleMap) and turned into surfaces; skeletal model generated from
the bones; attachment areas transferred from the msk-pipe atlas (itself derived from the
LHDL dissection data) by bone registration; all rotated into the OpenSim frame.

Licence: CC BY-NC-SA 2.0 BE (LICENSE).
