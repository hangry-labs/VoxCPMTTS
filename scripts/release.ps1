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

function Set-Utf8Text {
    param([string]$Path, [string]$Text)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($resolved, $Text, $encoding)
}

function Get-LineEndingAfter {
    param([string]$Content, [int]$StartIndex, [string]$Description)
    $lineFeedIndex = $Content.IndexOf("`n", $StartIndex)
    if ($lineFeedIndex -lt 0) {
        throw "$Description is not followed by a line ending."
    }
    if ($lineFeedIndex -gt 0 -and $Content[$lineFeedIndex - 1] -eq "`r") {
        return "`r`n"
    }
    return "`n"
}

function Convert-ToPackageVersion {
    param([string]$Version)
    if ($Version -match '^\d+\.\d+$') { return "$Version.0" }
    if ($Version -match '^\d+\.\d+\.\d+$') { return $Version }
    throw "Release version '$Version' must look like 1.0 or 1.0.0."
}

function Get-NextSnapshot {
    param([string]$Version)
    if ($Version -match '^(\d+)\.(\d+)(?:\.\d+)?$') {
        return "$($Matches[1]).$([int]$Matches[2] + 1)-snapshot"
    }
    throw "Cannot infer next snapshot from '$Version'. Pass NEXT_VERSION=..."
}

function Get-ProjectVersion {
    param([string]$Text)
    $match = [regex]::Match($Text, '(?m)^version = "([^"]+)"(?=\r?$)')
    if (-not $match.Success) {
        throw "pyproject.toml must contain one explicit project version."
    }
    return $match.Groups[1].Value
}

function Set-ProjectVersion {
    param([string]$Text, [string]$Version)
    return [regex]::Replace($Text, '(?m)^version = "[^"]+"(?=\r?$)', "version = `"$Version`"", 1)
}

function Get-SectionEnd {
    param([string]$Content, [int]$HeadingIndex)
    $nextHeading = [regex]::new('(?m)^#{2,3} ').Match($Content, $HeadingIndex + 1)
    if ($nextHeading.Success) { return $nextHeading.Index }
    return $Content.Length
}

function Update-DockerImageTags {
    param([string]$Text, [string]$Tag)
    return [regex]::Replace(
        $Text,
        'hangrylabs/voxcpmtts:(?:latest|v\d+\.\d+(?:\.\d+)?)(_tiny)?(?:@sha256:[0-9a-f]{64})?',
        { param($match) "hangrylabs/voxcpmtts:$Tag$($match.Groups[1].Value)" }
    )
}

function Get-ReleaseReadmeContent {
    param(
        [string]$Content,
        [string]$SnapshotHeading,
        [string]$StableHeading,
        [string]$ReleaseTag
    )
    $headingIndex = $Content.IndexOf($SnapshotHeading, [StringComparison]::Ordinal)
    if ($headingIndex -lt 0) {
        throw "README.md is missing the candidate heading '$SnapshotHeading'."
    }
    $sectionEnd = Get-SectionEnd $Content $headingIndex
    $section = $Content.Substring($headingIndex, $sectionEnd - $headingIndex)
    $developmentNotice = 'The current development snapshot is published through the rolling tags from `main`:'
    if (-not $section.Contains($developmentNotice)) {
        throw "README.md candidate section is missing the rolling-image notice."
    }

    $updatedSection = $section.Replace($SnapshotHeading, $StableHeading)
    $updatedSection = $updatedSection.Replace(
        $developmentNotice,
        "The ``$ReleaseTag`` release is available through the immutable version tags:"
    )
    $updatedSection = Update-DockerImageTags $updatedSection $ReleaseTag
    $updatedSection = $updatedSection.Replace(
        "No immutable ``$ReleaseTag`` image has been published yet. After publication and validation, this section must be updated with the exact ``$ReleaseTag@sha256:...`` and ``${ReleaseTag}_tiny@sha256:...`` Docker Hub references before the GitHub Release is announced.",
        "GitHub Actions publishes these tags from the annotated ``$ReleaseTag`` source tag. Add the verified Docker Hub top-level OCI index digests here before announcing the GitHub Release."
    )

    return $Content.Substring(0, $headingIndex) + $updatedSection + $Content.Substring($sectionEnd)
}

function Get-RollingDockerSection {
    param([string]$LineEnding)
    return @(
        'The current development snapshot is published through the rolling tags from `main`:',
        '',
        '**Full image**',
        '',
        '```bash',
        'docker run --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:latest',
        '```',
        '',
        '**Tiny image**',
        '',
        '```bash',
        'docker run --name voxcpmtts-tiny --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:latest_tiny',
        '```'
    ) -join $LineEnding
}

function Get-NextSnapshotReadmeContent {
    param(
        [string]$Content,
        [string]$StableHeading,
        [string]$NextDisplayVersion
    )
    $plannedHeading = "### v$NextDisplayVersion Planned"
    $snapshotHeading = "### v$NextDisplayVersion Snapshot"
    if ($Content.Contains($snapshotHeading)) { return $Content }

    $stableIndex = $Content.IndexOf($StableHeading, [StringComparison]::Ordinal)
    if ($stableIndex -lt 0) {
        throw "README.md is missing the promoted heading '$StableHeading'."
    }
    $lineEnding = Get-LineEndingAfter $Content $stableIndex "README.md heading '$StableHeading'"
    $rollingSection = Get-RollingDockerSection $lineEnding

    $plannedIndex = $Content.IndexOf($plannedHeading, [StringComparison]::Ordinal)
    if ($plannedIndex -ge 0) {
        if ($plannedIndex -gt $stableIndex) {
            throw "README.md '$plannedHeading' must appear before '$StableHeading'."
        }
        $plannedSection = $Content.Substring($plannedIndex, $stableIndex - $plannedIndex)
        $plannedSection = $plannedSection.Replace($plannedHeading, $snapshotHeading)
        $plannedSection = $plannedSection.TrimEnd("`r", "`n") +
            "$lineEnding$lineEnding$rollingSection$lineEnding$lineEnding"
        return $Content.Substring(0, $plannedIndex) + $plannedSection + $Content.Substring($stableIndex)
    }

    $newSection =
        "$snapshotHeading$lineEnding$lineEnding" +
        "- No changes yet.$lineEnding$lineEnding" +
        "$rollingSection$lineEnding$lineEnding"
    return $Content.Substring(0, $stableIndex) + $newSection + $Content.Substring($stableIndex)
}

function Invoke-Native {
    param([string]$Description, [scriptblock]$Action)
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$snapshotVersion = (Get-Content -Raw -Encoding utf8 -LiteralPath "VERSION").Trim()
$snapshotMatch = [regex]::Match($snapshotVersion, '^(\d+\.\d+(?:\.\d+)?)-snapshot$')
if (-not $snapshotMatch.Success) {
    throw "VERSION must be a snapshot version like 1.0-snapshot before release. Current: '$snapshotVersion'"
}

$releaseDisplayVersion = $snapshotMatch.Groups[1].Value
$releasePackageVersion = Convert-ToPackageVersion $releaseDisplayVersion
$releaseTag = "v$releaseDisplayVersion"
$snapshotPackageVersion = "$releasePackageVersion.dev0"
$nextSnapshotVersion = if ([string]::IsNullOrWhiteSpace($NextVersion)) {
    Get-NextSnapshot $releaseDisplayVersion
} else {
    $NextVersion.Trim()
}
$nextMatch = [regex]::Match($nextSnapshotVersion, '^(\d+\.\d+(?:\.\d+)?)-snapshot$')
if (-not $nextMatch.Success) {
    throw "NextVersion must look like 1.1-snapshot or 1.1.0-snapshot. Current: '$nextSnapshotVersion'"
}
$nextDisplayVersion = $nextMatch.Groups[1].Value
$nextReleaseVersion = Convert-ToPackageVersion $nextDisplayVersion
if ([version]$nextReleaseVersion -le [version]$releasePackageVersion) {
    throw "NextVersion '$nextSnapshotVersion' must be newer than '$releaseDisplayVersion'."
}
$nextPackageVersion = "$nextReleaseVersion.dev0"

$pyproject = Get-Content -Raw -Encoding utf8 -LiteralPath "pyproject.toml"
if ((Get-ProjectVersion $pyproject) -ne $snapshotPackageVersion) {
    throw "pyproject.toml version must be '$snapshotPackageVersion' for VERSION '$snapshotVersion'."
}

$snapshotHeading = "### v$releaseDisplayVersion Snapshot"
$stableHeading = "### $releaseTag"
$readme = Get-Content -Raw -Encoding utf8 -LiteralPath "README.md"
$releaseReadme = Get-ReleaseReadmeContent $readme $snapshotHeading $stableHeading $releaseTag
$nextReadme = Get-NextSnapshotReadmeContent $releaseReadme $stableHeading $nextDisplayVersion
$releasePyproject = Set-ProjectVersion $pyproject $releasePackageVersion
$nextPyproject = Set-ProjectVersion $releasePyproject $nextPackageVersion

$dockerHub = Get-Content -Raw -Encoding utf8 -LiteralPath "docs/dockerhub.md"
$releaseDockerHub = Update-DockerImageTags $dockerHub $releaseTag
$releaseDockerHub = $releaseDockerHub.Replace(
    "Snapshot tags remain intentionally unpinned because they move with ``main``. No immutable ``$releaseTag`` image has been published yet.",
    "The ``$releaseTag`` and ``${releaseTag}_tiny`` tags are published from the annotated source tag. Add verified top-level OCI index digests after publication."
)

if ((Get-ProjectVersion $releasePyproject) -ne $releasePackageVersion) {
    throw "Release transformation did not produce package version '$releasePackageVersion'."
}
if ((Get-ProjectVersion $nextPyproject) -ne $nextPackageVersion) {
    throw "Next-snapshot transformation did not produce package version '$nextPackageVersion'."
}
if (-not $nextReadme.Contains("### v$nextDisplayVersion Snapshot") -or
    -not $nextReadme.Contains('hangrylabs/voxcpmtts:latest') -or
    -not $nextReadme.Contains($stableHeading)) {
    throw "README.md next-snapshot transformation is incomplete."
}

$branch = (git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not determine the current Git branch." }
if ($branch -ne "main") { throw "Releases must run from main. Current branch: '$branch'" }

$status = git status --porcelain --untracked-files=all -- . ":(exclude).ai" ":(exclude).ai/**" ":(exclude)todo" ":(exclude)todo/**"
if ($LASTEXITCODE -ne 0) { throw "Could not inspect the Git working tree." }
if ($status) {
    if (Test-Enabled $DryRun) {
        Write-Warning "The real release requires a clean working tree outside .ai/ and todo/."
        $status | ForEach-Object { Write-Host "  $_" }
    } else {
        throw "Working tree outside .ai/ and todo/ must be clean before release.`n$($status -join "`n")"
    }
}

if (-not (Test-Enabled $DryRun)) {
    Invoke-Native "Fetch origin/main and tags" { git fetch origin main --tags }
    $head = (git rev-parse HEAD).Trim()
    $originMain = (git rev-parse refs/remotes/origin/main).Trim()
    if ($head -ne $originMain) {
        throw "main must be synchronized with origin/main before release. HEAD=$head origin/main=$originMain"
    }
}

if (git tag --list $releaseTag) {
    throw "Tag $releaseTag already exists."
}

Write-Host "Release version: $releaseDisplayVersion"
Write-Host "Release tag:     $releaseTag"
Write-Host "Package version: $releasePackageVersion"
Write-Host "Next snapshot:   $nextSnapshotVersion"
Write-Host "Next package:    $nextPackageVersion"
Write-Host "==> Release and next-snapshot transformations validated in memory"

if (Test-Enabled $DryRun) {
    Write-Host "Dry run complete. No files, builds, commits, tags, or remote refs were changed."
    exit 0
}

Write-Host "==> Update files for $releaseTag"
Set-Utf8Text "VERSION" "$releaseDisplayVersion`n"
Set-Utf8Text "pyproject.toml" $releasePyproject
Set-Utf8Text "README.md" $releaseReadme
Set-Utf8Text "docs/dockerhub.md" $releaseDockerHub

if (-not (Test-Enabled $SkipValidation)) {
    Write-Host "==> Run release validation"
    Invoke-Native "Git whitespace validation" { git diff --check }
    Invoke-Native "Dockerfile validation" { docker build --check . }
    Invoke-Native "Python compilation" { python -m compileall -q voxcpm scripts }
    Invoke-Native "Tiny image build" { task image-tiny }
    Invoke-Native "Dockerized test suite" { task test }
    Invoke-Native "Baked image build" { task image }
}

Write-Host "==> Commit and tag $releaseTag"
Invoke-Native "Stage release metadata" { git add -- VERSION pyproject.toml README.md docs/dockerhub.md }
Invoke-Native "Create release commit" { git commit -m "release: $releaseTag" }
Invoke-Native "Create annotated release tag" { git tag -a $releaseTag -m "Release $releaseTag" }

Write-Host "==> Prepare $nextSnapshotVersion"
Set-Utf8Text "VERSION" "$nextSnapshotVersion`n"
Set-Utf8Text "pyproject.toml" $nextPyproject
Set-Utf8Text "README.md" $nextReadme
Invoke-Native "Stage next-snapshot metadata" { git add -- VERSION pyproject.toml README.md }
Invoke-Native "Create next-snapshot commit" { git commit -m "chore: start $nextSnapshotVersion" }

Write-Host "Release workflow complete. Commits and tag are local. Publish atomically with:"
Write-Host "  git push --atomic origin main $releaseTag"
