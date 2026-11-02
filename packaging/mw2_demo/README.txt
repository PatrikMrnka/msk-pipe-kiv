Muscle Wrapping 2.x demo for msk-pipe (Windows x64)
===================================================

Muscle Wrapping 2.x (OsimMuscleGeneratorTool, a plugin of OpenSim 4.5) turns the output of
msk-pipe (06_mw2_input: model, muscle surfaces, attachment areas) into muscle fibres and
their lengths and moment arms over the motion. It is NOT part of msk-pipe; this package
only lets you try the result. Nothing has to be installed.

Run
  Double-click run_demo.cmd          the bundled example (example\, LHDL CT)
  run_demo.cmd <run folder>          a msk-pipe run (or its 06_mw2_input folder)
  run_demo.cmd <run folder> <work>   own work folder

The input is copied to output\<run name>\ (or <work>) and Muscle Wrapping runs there
(a few minutes on the example):
  bone_model_mw2.osim             OpenSim model, each muscle as its fibres
  analysis\<muscle>_Lengths.sto   fibre lengths over the motion
  analysis\<muscle>_MomentArms_*  moment arms per coordinate
  mw2.log                         log of Muscle Wrapping

Open bone_model_mw2.osim in the OpenSim GUI (4.5) to view the fibres.

Muscle Wrapping 2.x occasionally crashes (access violation, about 1 run in 10). The demo then
runs it again, at most 3 times (set MW2_DEMO_ATTEMPTS=5 for more); the log of a crashed
attempt is kept as mw2_crash<n>.log. A successful run gives the same result.

Licences
  Muscle Wrapping 2.x: Apache-2.0, Copyright 2018 University of West Bohemia (LICENSE).
  OpenSim, Simbody, VTK and the other libraries in bin\: THIRD_PARTY_LICENSES.md, licenses\.
  The example (example\) is derived from the LHDL dataset: CC BY-NC-SA 2.0 BE
  (example\LICENSE), not Apache-2.0.

Cite
  Modenese L., Kohout J. (2020). Automated Generation of Three-Dimensional Complex Muscle
  Geometries for Use in Personalised Musculoskeletal Models. Annals of Biomedical
  Engineering 48, 1793-1804. https://doi.org/10.1007/s10439-020-02490-4

https://github.com/kivzcu/msk-pipe
