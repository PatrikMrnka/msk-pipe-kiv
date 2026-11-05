# Experiments of the paper

Validation experiments behind the numbers in the manuscript. The data (LHDL, TLEM 2.0) are not in the repository; paths point to the local `data\` folder (see the comments in each file).

| ID | What it measures | Files |
|---|---|---|
| E13 | Timings and memory of the whole pipeline, CPU and GPU, n = 3, no cache | `e13_timing.yaml` |
| E14 | Muscle Wrapping 2.x input: exported muscles, attachment checks, attachment–muscle gaps | reference run + `mskpipe stats` |
| E16 | Mean fibre length over hip flexion vs the original Muscle Wrapping run of LHDL | reference run + `tools/fibre_lengths.py` |
| E18 | Bones, muscles and attachment areas vs ground truth (LHDL, TLEM 2.0; CT, MRI) | `e18_gt.yaml`, `gt/*.yaml`, `tools/gt_compare.py` |
| E19 | Right atlas mirrored to the left LHDL leg, paired with the left atlas | `e19_mirror.yaml`, `tools/attachment_checks.py` |

The reference run is LHDL CT with the default settings and the release atlas `lhdl-v4`. Muscle Wrapping 2.x is run on its `06_mw2_input` outside the pipeline.

## Run

From the repository root (Windows PowerShell):

```powershell
.\paper\experiments\run_experiments.ps1 -DryRun     # plan only
.\paper\experiments\run_experiments.ps1             # E14, E16, E18, E19, E13
.\paper\experiments\run_experiments.ps1 -Only E18,E19
```

Outputs go to `..\paper\stats\v<version>\<experiment>` with `experiments.json` (mskpipe version, git, batch and run folders) and a transcript. Options: `-Run` (reuse a reference run), `-Mw2Bin` (Muscle Wrapping build, default `..\MW2-MASTER\Build\Bin`), `-Mw2Reference` (folder with the reference `*_Lengths.sto`), `-Mw2Reference2` (a second run of the reference: run-to-run noise of Muscle Wrapping), `-Out`.

E13 measures time: run it on an otherwise idle computer.
