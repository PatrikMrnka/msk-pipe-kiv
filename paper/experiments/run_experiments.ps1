<#
.SYNOPSIS
    Runs the validation experiments of the paper and collects their statistics.

.DESCRIPTION
    E14/E16  reference run (LHDL CT, default settings) -> Muscle Wrapping 2.x ->
             export checks (mskpipe stats) and fibre lengths vs the original
             Muscle Wrapping run of LHDL (tools/fibre_lengths.py: metrics, curves with
             the 5-95 % band of the fibres, violins; -Mw2Reference2 = second run of the
             reference for the run-to-run noise of Muscle Wrapping)
    E18      pipeline vs ground truth, LHDL + TLEM 2.0, CT + MRI (e18_gt.yaml, gt_compare)
    E19      right atlas mirrored to the left LHDL leg (e19_mirror.yaml, attachment_checks pair)
    E13      timings CPU + GPU, n = 3, no cache (e13_timing.yaml); run it on an idle PC

    Everything goes to -Out (default ..\paper\stats\v<version>\<experiment>), with a
    transcript and experiments.json (versions, batch folders, run folders).
    Data are not in the repository: paths point to the local data\ folder.

.EXAMPLE
    .\paper\experiments\run_experiments.ps1 -DryRun
.EXAMPLE
    .\paper\experiments\run_experiments.ps1 -Only E14,E16 -Run runs\20261104-214240_Cadaver1_LowerLimbsCT_e415ba1f
.EXAMPLE
    .\paper\experiments\run_experiments.ps1 -Only E18,E19
.EXAMPLE
    .\paper\experiments\run_experiments.ps1 -Only E18,E19 -E18Batch runs\batches\20261104-221232_E18
#>
[CmdletBinding()]
param(
    [ValidateSet("E13", "E14", "E16", "E18", "E19")]
    [string[]]$Only = @("E14", "E16", "E18", "E19", "E13"),
    [string]$Out = "",
    [string]$Run = "",
    [string]$E18Batch = "",   # finished E18 batch: only stats + gt_compare, no new runs
    [string]$E19Batch = "",   # finished E19 batch: only gt_compare, pair, gt-bones, fibre-hull
    [string]$Mw2Bin = "..\MW2-MASTER\Build\Bin",
    [string]$Mw2Reference = "..\reference\mw2_baseline\Experiments_luca_all",
    [string]$Mw2ReferenceMotion = "",
    [string]$Mw2Reference2 = "",
    [int]$Mw2Attempts = 3,
    [switch]$ForceMw2,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$LhdlCt = "data\reference-GT\lhdl_msf\volumes\Cadaver1_LowerLimbsCT.nii.gz"
$AtlasV4 = "data/reference-GT/lhdl_atlas_v4"

function Invoke-Native([string]$What, [scriptblock]$Command) {
    Write-Host ""
    Write-Host "=== $What" -ForegroundColor Cyan
    & $Command | Out-Host   # Out-Host: native output also goes to the transcript
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Get-NewFolder([string]$Parent, [string]$Filter, [datetime]$Since) {
    $hit = Get-ChildItem $Parent -Directory -Filter $Filter -ErrorAction SilentlyContinue |
        Where-Object { $_.CreationTime -ge $Since } | Sort-Object Name | Select-Object -Last 1
    if (-not $hit) { throw "No new folder $Parent\$Filter since $Since" }
    return $hit.FullName
}

function Invoke-Batch([string]$Experiment, [string]$Yaml) {
    $since = Get-Date
    Invoke-Native "$Experiment batch ($Yaml)" { pixi run -e gpu mskpipe batch $Yaml }
    return Get-NewFolder "runs\batches" "*_$Experiment*" $since
}

function Invoke-Mw2([string]$RunDir) {
    $in = Join-Path $RunDir "06_mw2_input"
    $records = (Get-Content (Join-Path $in "export.json") -Raw | ConvertFrom-Json).muscles
    $n = @($records | Where-Object { $_.status -ne "excluded" }).Count
    $analysis = Join-Path $in "analysis"
    $have = @(Get-ChildItem $analysis -Filter "*_Lengths.sto" -ErrorAction SilentlyContinue).Count
    if ($have -eq $n -and -not $ForceMw2) {
        Write-Host "Muscle Wrapping output complete ($have/$n muscles), not run again (-ForceMw2)"
        return
    }
    $bin = (Resolve-Path $Mw2Bin).Path
    $exe = Join-Path $bin "opensim-cmd.exe"
    $dll = Join-Path $bin "OsimMuscleGeneratorTool.dll"
    $oldPath = $env:PATH
    $env:PATH = "$bin;$env:PATH"   # the opensim-cmd of another OpenSim fails to load the plugin
    Push-Location $in
    try {
        for ($i = 1; $i -le $Mw2Attempts; $i++) {
            if (Test-Path analysis) { Remove-Item analysis -Recurse -Force }
            Write-Host "Muscle Wrapping attempt $i/$Mw2Attempts ($n muscles) ..."
            $ErrorActionPreference = "Continue"   # MW2 and VTK write to stderr
            & $exe -L $dll run-tool setup_MuscleGeneratorTool.xml *>&1 | Out-File -Encoding utf8 mw2_all.log
            $code = $LASTEXITCODE
            $ErrorActionPreference = "Stop"
            $have = @(Get-ChildItem analysis -Filter "*_Lengths.sto" -ErrorAction SilentlyContinue).Count
            if ($code -eq 0 -and $have -eq $n) { break }
            Copy-Item mw2_all.log "mw2_crash$i.log" -Force
            Write-Warning "Muscle Wrapping attempt ${i}: exit code $code, $have/$n length files"
        }
    } finally {
        Pop-Location
        $env:PATH = $oldPath
    }
    if ($code -ne 0 -or $have -ne $n) { throw "Muscle Wrapping failed $Mw2Attempts times (logs mw2_crash*.log in $in)" }
    Write-Host "Muscle Wrapping OK: $have/$n muscles"
}

# --------------------------------------------------------------------------- setup
$version = (@(pixi run -e gpu mskpipe --version)[-1] -replace "^mskpipe\s+", "").Trim()
if ($LASTEXITCODE -ne 0) { throw "mskpipe is not installed in the pixi environment gpu" }
$describe = git describe --tags --dirty --always
if ($version -match "dev|\+") {
    Write-Warning "mskpipe $version is not a release (git: $describe). For the paper: git switch --detach v<version>; pixi reinstall -e gpu mskpipe; pixi reinstall -e dev-cpu mskpipe"
}
if (-not $Out) { $Out = "..\paper\stats\v$version" }

if ($DryRun) {
    foreach ($e in @(@("E18", "e18_gt"), @("E19", "e19_mirror"), @("E13", "e13_timing"))) {
        if ($Only -contains $e[0]) {
            Invoke-Native "$($e[0]) plan" { pixi run -e gpu mskpipe batch "paper/experiments/$($e[1]).yaml" --dry-run }
        }
    }
    Write-Host ""
    Write-Host "mskpipe $version (git $describe); outputs would go to $Out"
    exit 0
}

New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
Start-Transcript -Path (Join-Path $Out "run_experiments_$(Get-Date -Format yyyyMMdd-HHmmss).log") -Append | Out-Null
$info = [ordered]@{
    mskpipe = $version
    git = $describe
    started = (Get-Date).ToString("s")
    computer = $env:COMPUTERNAME
    experiments = [ordered]@{}
}
$infoPath = Join-Path $Out "experiments.json"
if (Test-Path $infoPath) {
    $old = Get-Content $infoPath -Raw | ConvertFrom-Json
    if ($old.experiments) {
        foreach ($p in $old.experiments.PSObject.Properties) { $info.experiments[$p.Name] = $p.Value }
    }
}
function Save-Info { $info | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 $infoPath }

try {
    # ----------------------------------------------------------------------- E14, E16
    if ($Only -contains "E14" -or $Only -contains "E16") {
        if (-not $Run) {
            $since = Get-Date
            Invoke-Native "E14/E16 reference run (LHDL CT, default settings, atlas lhdl-v4)" {
                pixi run -e gpu mskpipe run $LhdlCt -m ct --set "attachments.params.atlas_dir=$AtlasV4"
            }
            $Run = Get-NewFolder "runs" "*_Cadaver1_LowerLimbsCT_*" $since
        }
        $Run = (Resolve-Path $Run).Path
        Write-Host "Reference run: $Run"
        Invoke-Mw2 $Run
        if ($Only -contains "E14") {
            $o = Join-Path $Out "e14"
            Invoke-Native "E14 export checks" { pixi run -e dev-cpu mskpipe stats $Run -o $o }
            Copy-Item (Join-Path $Run "06_mw2_input\export.json"), (Join-Path $Run "06_mw2_input\mw2_all.log") $o -Force
            $info.experiments["E14"] = [ordered]@{ run = $Run; out = $o }
        }
        if ($Only -contains "E16") {
            $o = Join-Path $Out "e16"
            $refArgs = @("--ref", $Mw2Reference)
            if ($Mw2ReferenceMotion) { $refArgs += @("--ref-motion", $Mw2ReferenceMotion) }
            if ($Mw2Reference2) { $refArgs += @("--ref2", $Mw2Reference2) }   # Muscle Wrapping noise
            Invoke-Native "E16 fibre lengths vs the original Muscle Wrapping run" {
                pixi run -e dev-cpu python tools/fibre_lengths.py $Run @refArgs --out $o
            }
            $info.experiments["E16"] = [ordered]@{ run = $Run; reference = (Resolve-Path $Mw2Reference).Path; out = $o }
        }
        Save-Info
    }

    # ----------------------------------------------------------------------- E18
    $o18 = Join-Path $Out "e18"
    if ($Only -contains "E18") {
        if ($E18Batch) { $b18 = (Resolve-Path $E18Batch).Path }
        else { $b18 = Invoke-Batch "E18" "paper/experiments/e18_gt.yaml" }
        Invoke-Native "E18 stats" { pixi run -e dev-cpu mskpipe stats $b18 -e E18 -o $o18 }
        foreach ($g in "lhdl", "lhdl_dissection", "tlem2_ct", "tlem2_mri") {
            Invoke-Native "E18 gt_compare $g (mesh)" {
                pixi run -e dev-cpu python tools/gt_compare.py --gt "paper/experiments/gt/$g.yaml" --out $o18 $b18
            }
        }
        foreach ($g in "lhdl", "tlem2_mri") {
            Invoke-Native "E18 gt_compare $g (mw2)" {
                pixi run -e dev-cpu python tools/gt_compare.py --gt "paper/experiments/gt/$g.yaml" --stage mw2 --out $o18 $b18
            }
        }
        $info.experiments["E18"] = [ordered]@{ batch = $b18; out = $o18 }
        Save-Info
    }

    # ----------------------------------------------------------------------- E19
    if ($Only -contains "E19") {
        $base = Join-Path $o18 "attachments_LHDL_mesh.csv"
        if (-not (Test-Path $base)) { throw "E19 needs the E18 results ($base): run -Only E18 first" }
        if (-not (Test-Path "data\reference-GT\lhdl_atlas_mirror\atlas.json")) {
            Invoke-Native "E19 mirrored atlas" { pixi run -e dev-cpu python tools/attachment_checks.py mirror-atlas }
        }
        $o19 = Join-Path $Out "e19"
        if ($E19Batch) { $b19 = (Resolve-Path $E19Batch).Path }
        else { $b19 = Invoke-Batch "E19" "paper/experiments/e19_mirror.yaml" }
        Invoke-Native "E19 gt_compare" {
            pixi run -e dev-cpu python tools/gt_compare.py --gt paper/experiments/gt/lhdl.yaml --out "$o19\runs" $b19
        }
        Invoke-Native "E19 pair (mirrored atlas vs left atlas, per area)" {
            pixi run -e dev-cpu python tools/attachment_checks.py pair --base $base --mirror "$o19\runs\attachments_LHDL_mesh.csv" --out "$o19\pair"
        }
        Invoke-Native "E19 mirrored atlas on the GT bones" {
            pixi run -e dev-cpu python tools/attachment_checks.py gt-bones --out "$o19\gt_bones"
        }
        Invoke-Native "E19 definition floor (fibre hull + 4 mm)" {
            pixi run -e dev-cpu python tools/attachment_checks.py fibre-hull --radius 2 4 8 --out "$o19\fibre_hull"
        }
        $info.experiments["E19"] = [ordered]@{ batch = $b19; base = $base; out = $o19 }
        Save-Info
    }

    # ----------------------------------------------------------------------- E13
    if ($Only -contains "E13") {
        Write-Host ""
        Write-Host "E13 measures timings: do not use the PC until it finishes (~70 min)." -ForegroundColor Yellow
        $o13 = Join-Path $Out "e13"
        $b13 = Invoke-Batch "E13" "paper/experiments/e13_timing.yaml"
        Invoke-Native "E13 stats" { pixi run -e dev-cpu mskpipe stats $b13 -e E13 -o $o13 }
        $info.experiments["E13"] = [ordered]@{ batch = $b13; out = $o13 }
        Save-Info
    }
} finally {
    $info["finished"] = (Get-Date).ToString("s")
    Save-Info
    Stop-Transcript | Out-Null
}
Write-Host ""
Write-Host "Done: $Out" -ForegroundColor Green
