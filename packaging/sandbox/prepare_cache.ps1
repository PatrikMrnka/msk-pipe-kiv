# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
  Prepare C:\msk-pipe-sandbox-cache for msk-pipe-cached.wsb (repeated sandbox tests
  without downloading GPU support and model weights again).

.DESCRIPTION
  Creates the folders the sandbox maps (Windows Sandbox does not start when a mapped host
  folder is missing). With -FromHost it fills them from this computer, where msk-pipe has
  already downloaded everything:
    %LOCALAPPDATA%\msk-pipe         -> msk-pipe          (GPU support, MuscleMap, atlas)
    %USERPROFILE%\.totalsegmentator -> totalsegmentator  (weights, licence key)
  It is a folder of its own (not inside C:\msk-pipe-sandbox, which is cleaned before each
  test release).
  Otherwise the first sandbox session downloads them once into the cache.

.PARAMETER FromHost
  Copy the data of this computer into the cache (robocopy, only what is missing/newer).
.PARAMETER NoLicence
  Leave out the TotalSegmentator licence key (to test the Setup window's key dialog).
.PARAMETER Clear
  Delete the cache first.

.EXAMPLE
  packaging\sandbox\prepare_cache.ps1 -FromHost
  Start-Process packaging\sandbox\msk-pipe-cached.wsb
#>
param(
  [switch]$FromHost,
  [switch]$NoLicence,
  [switch]$Clear,
  [string]$Cache = "C:\msk-pipe-sandbox-cache"
)
$ErrorActionPreference = "Stop"
$cache = $Cache
$mskDst = Join-Path $cache "msk-pipe"
$tsDst = Join-Path $cache "totalsegmentator"

if ($Clear -and (Test-Path $cache)) { Remove-Item $cache -Recurse -Force }
New-Item -ItemType Directory -Force "C:\msk-pipe-sandbox", $mskDst, $tsDst | Out-Null

function Copy-Tree($src, $dst, [string[]]$excludeDirs = @(), [string[]]$excludeFiles = @()) {
  if (-not (Test-Path $src)) { Write-Host "  (not on this computer: $src)"; return }
  $rcArgs = @($src, $dst, "/E", "/NFL", "/NDL", "/NJH", "/NP", "/R:1", "/W:1")
  if ($excludeDirs.Count) { $rcArgs += "/XD"; $rcArgs += $excludeDirs }
  if ($excludeFiles.Count) { $rcArgs += "/XF"; $rcArgs += $excludeFiles }
  & robocopy @rcArgs | Out-Host
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $src" }
}

if ($FromHost) {
  Write-Host "msk-pipe data -> $mskDst"
  # locks and download leftovers are per machine/session
  Copy-Tree (Join-Path $env:LOCALAPPDATA "msk-pipe") $mskDst @("locks", "downloads", "*.partial", "*.old")
  Write-Host "TotalSegmentator -> $tsDst"
  $skip = if ($NoLicence) { @("config.json") } else { @() }
  Copy-Tree (Join-Path $env:USERPROFILE ".totalsegmentator") $tsDst @() $skip
}

$size = (Get-ChildItem $cache -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
Write-Host ("Cache ready: {0} ({1:N1} GB)" -f $cache, ($size / 1GB))
Write-Host "Start: Start-Process $(Join-Path $PSScriptRoot 'msk-pipe-cached.wsb')"
