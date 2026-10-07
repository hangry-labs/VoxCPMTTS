param(
    [string]$DryRun = "0",
    [string]$NextVersion = "",
    [string]$SkipValidation = "0"
)

$ErrorActionPreference = "Stop"

function Test-Enabled {
    param([string]$Value)
    return $Value -match '^(1|true|yes|y)$'
}

function Set-Text {
    param(
        [string]$Path,
        [string]$Text
    )
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($resolved, $Text, $encoding)
}

function Convert-ToPackageVersion {
    param([string]$Version)
    if ($Version -match '^\d+\.\d+$') {
        return "$Version.0"
    }
    if ($Version -match '^\d+\.\d+\.\d+$') {
        return $Version
    }
    throw "Release version '$Version' must look like 1.0 or 1.0.0."
}

function Get-NextSnapshot {
    param([string]$Version)
    if ($Version -match '^(\d+)\.(\d+)$') {
        return "$($Matches[1]).$([int]$Matches[2] + 1)-snapshot"
    }
    if ($Version -match '^(\d+)\.(\d+)\.(\d+)$') {
        return "$($Matches[1]).$([int]$Matches[2] + 1)-snapshot"
    }
    throw "Cannot infer next snapshot from '$Version'. Pass NEXT_VERSION=..."
}

function Get-ProjectVersion {
    param([string]$Text)
    $match = [regex]::Match($Text, '(?m)^version = "([^"]+)"$')
    if (-not $match.Success) {
        throw "pyproject.toml must contain one explicit project version."
    }
    return $match.Groups[1].Value
}

function Set-ProjectVersion {
    param(
        [string]$Text,
        [string]$Version
    )
    return [regex]::Replace($Text, '(?m)^version = "[^"]+"$', "version = `"$Version`"")
}

function Update-DockerImageTags {
    param(
        [string]$Text,
        [string]$ReleaseTag
    )
    return [regex]::Replace(
        $Text,
        'hangrylabs/voxcpmtts:(?:latest|v\d+\.\d+(?:\.\d+)?)(_tiny)?(?:@sha256:[0-9a-f]{64})?',
        { param($match) "hangrylabs/voxcpmtts:$ReleaseTag$($match.Groups[1].Value)" }
    )
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if (-not (Test-Path -LiteralPath "VERSION")) {
    throw "VERSION file is missing from repo root."
}

$snapshotVersion = (Get-Content -Raw -LiteralPath "VERSION").Trim()
if ($snapshotVersion -notmatch '^(\d+\.\d+(?:\.\d+)?)-snapshot$') {
    throw "VERSION must be a snapshot version like 1.0-snapshot before release. Current: '$snapshotVersion'"
}

$releaseVersion = $Matches[1]
$releaseTag = "v$releaseVersion"
$releasePackageVersion = Convert-ToPackageVersion $releaseVersion
$snapshotPackageVersion = "$releasePackageVersion.dev0"

if ([string]::IsNullOrWhiteSpace($NextVersion)) {
    $nextSnapshotVersion = Get-NextSnapshot $releaseVersion
} else {
    $nextSnapshotVersion = $NextVersion.Trim()
}

if ($nextSnapshotVersion -notmatch '^\d+\.\d+(?:\.\d+)?-snapshot$') {
    throw "NextVersion must look like 1.1-snapshot or 1.1.0-snapshot. Current: '$nextSnapshotVersion'"
}

$nextReleaseBase = $nextSnapshotVersion -replace '-snapshot$', ''
$nextPackageVersion = "$(Convert-ToPackageVersion $nextReleaseBase).dev0"

$pyproject = Get-Content -Raw -LiteralPath "pyproject.toml"
$currentPackageVersion = Get-ProjectVersion $pyproject
if ($currentPackageVersion -ne $snapshotPackageVersion) {
    throw "pyproject.toml version must be '$snapshotPackageVersion' for VERSION '$snapshotVersion'. Current: '$currentPackageVersion'"
}

$readme = Get-Content -Raw -LiteralPath "README.md"
$snapshotHeading = "### v$releaseVersion Snapshot"
if (-not $readme.Contains($snapshotHeading)) {
    throw "README.md is missing the candidate heading '$snapshotHeading'."
}

$releasePyproject = Set-ProjectVersion $pyproject $releasePackageVersion
$releaseReadme = $readme.Replace($snapshotHeading, "### $releaseTag")
$releaseReadme = $releaseReadme.Replace(
    "The current development snapshot is published through the rolling tags from ``main``:",
    "The ``$releaseTag`` release is available through the immutable version tags:"
)
$releaseReadme = Update-DockerImageTags $releaseReadme $releaseTag

$dockerHub = Get-Content -Raw -LiteralPath "docs/dockerhub.md"
$releaseDockerHub = Update-DockerImageTags $dockerHub $releaseTag

if ($releaseReadme.Contains($snapshotHeading) -or -not $releaseReadme.Contains("### $releaseTag")) {
    throw "README.md release transformation did not promote the candidate heading."
}
if ((Get-ProjectVersion $releasePyproject) -ne $releasePackageVersion) {
    throw "pyproject.toml release transformation did not produce '$releasePackageVersion'."
}

$status = git status --porcelain -- . ":(exclude)todo" ":(exclude).ai"
if ($status -and -not (Test-Enabled $DryRun)) {
    throw "Working tree outside .ai/ and todo/ must be clean before release. Commit or stash release-relevant changes first."
}

if (git rev-parse -q --verify "refs/tags/$releaseTag" 2>$null) {
    throw "Tag $releaseTag already exists."
}

Write-Host "Release version: $releaseVersion"
Write-Host "Release tag:     $releaseTag"
Write-Host "Package version: $releasePackageVersion"
Write-Host "Next snapshot:   $nextSnapshotVersion"
Write-Host "Next package:    $nextPackageVersion"
Write-Host "==> Release document and version transformations validated in memory"

if (Test-Enabled $DryRun) {
    Write-Host "Dry run only: no files, builds, commits, or tags were changed."
    exit 0
}

Write-Host "==> Update files for $releaseTag"
Set-Text "VERSION" $releaseVersion
Set-Text "pyproject.toml" $releasePyproject
Set-Text "README.md" $releaseReadme
Set-Text "docs/dockerhub.md" $releaseDockerHub

if (-not (Test-Enabled $SkipValidation)) {
    Write-Host "==> Run release validation"
    python -m compileall -q voxcpm scripts
    task image-tiny
    task test
    task image
}

Write-Host "==> Commit and tag $releaseTag"
git add VERSION pyproject.toml README.md docs/dockerhub.md
git commit -m "release: $releaseTag"
git tag -a $releaseTag -m "Release $releaseTag"

Write-Host "==> Prepare $nextSnapshotVersion"
Set-Text "VERSION" $nextSnapshotVersion
$nextPyproject = Set-ProjectVersion $releasePyproject $nextPackageVersion
Set-Text "pyproject.toml" $nextPyproject
git add VERSION pyproject.toml
git commit -m "chore: start $nextSnapshotVersion"

Write-Host "Release workflow complete. Commits and tag are local. Publish with:"
Write-Host "  git push origin main"
Write-Host "  git push origin $releaseTag"
