import { t } from './i18n.js'

const REPOSITORY = 'Hangry-Labs/VoxCPMTTS'
const CACHE_KEY = 'voxcpmtts-version-check-v2'
const CACHE_MAX_AGE_MS = 60 * 60 * 1000
const RELEASES_URL = `https://github.com/${REPOSITORY}/releases`
const SNAPSHOT_BUILDS_URL = `https://github.com/${REPOSITORY}/actions/workflows/docker-build.yml`

function readCache() {
  try {
    const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null')
    return cached && Date.now() - cached.checkedAt < CACHE_MAX_AGE_MS ? cached.data : null
  } catch {
    return null
  }
}

function writeCache(data) {
  try { localStorage.setItem(CACHE_KEY, JSON.stringify({ checkedAt: Date.now(), data })) } catch {}
}

function parseVersion(value) {
  const match = String(value || '').trim().replace(/^v/i, '').match(/^(\d+)\.(\d+)(?:\.(\d+))?/)
  return match ? match.slice(1).map((part) => Number(part || 0)) : null
}

function compareVersions(left, right) {
  if (!left || !right) return 0
  for (let index = 0; index < 3; index += 1) {
    if (left[index] !== right[index]) return left[index] > right[index] ? 1 : -1
  }
  return 0
}

function isSnapshot(version) {
  return /(?:snapshot|dev|nightly|alpha|beta|rc)/i.test(String(version || ''))
}

async function fetchGithubJson(path, fetchImpl) {
  const response = await fetchImpl(`https://api.github.com/repos/${REPOSITORY}/${path}`, {
    headers: { Accept: 'application/vnd.github+json' },
  })
  if (!response.ok) return null
  return response.json()
}

async function loadRemoteVersion(fetchImpl) {
  const cached = readCache()
  if (cached) return cached
  const [releaseResult, snapshotResult] = await Promise.allSettled([
    fetchGithubJson('releases?per_page=10', fetchImpl),
    fetchGithubJson('actions/workflows/docker-build.yml/runs?branch=main&status=success&per_page=1', fetchImpl),
  ])
  const releases = releaseResult.status === 'fulfilled' && Array.isArray(releaseResult.value) ? releaseResult.value : []
  const release = releases.find((candidate) => !candidate.draft && !candidate.prerelease) || null
  const snapshot = snapshotResult.status === 'fulfilled' ? snapshotResult.value?.workflow_runs?.[0] : null
  const data = {
    release: release ? {
      version: release.tag_name || release.name || '',
      url: release.html_url || RELEASES_URL,
    } : null,
    snapshot: snapshot ? {
      revision: snapshot.head_sha || '',
      date: snapshot.updated_at || snapshot.run_started_at || '',
      url: snapshot.html_url || SNAPSHOT_BUILDS_URL,
    } : null,
  }
  if (data.release || data.snapshot) writeCache(data)
  return data
}

export function determineUpdate(localStatus, remote) {
  const localVersion = parseVersion(localStatus?.version)
  const releaseVersion = parseVersion(remote?.release?.version)
  const snapshot = isSnapshot(localStatus?.version)
  const releaseComparison = compareVersions(releaseVersion, localVersion)

  if (releaseVersion && (releaseComparison > 0 || (snapshot && releaseComparison === 0))) {
    return { type: 'release', url: remote.release.url || RELEASES_URL }
  }
  if (!snapshot || !remote?.snapshot) return null

  const localRevision = String(localStatus?.revision || '').toLowerCase()
  const remoteRevision = String(remote.snapshot.revision || '').toLowerCase()
  if (localRevision && remoteRevision.startsWith(localRevision)) return null
  const localDate = Date.parse(localStatus?.build_date || '')
  const remoteDate = Date.parse(remote.snapshot.date || '')
  if (Number.isFinite(localDate) && Number.isFinite(remoteDate) && remoteDate > localDate) {
    return { type: 'snapshot', url: remote.snapshot.url || SNAPSHOT_BUILDS_URL }
  }
  return null
}

export class VersionCheck {
  constructor(element, { fetchImpl = fetch } = {}) {
    this.element = element
    this.fetchImpl = fetchImpl
  }

  async check(localStatus) {
    try {
      const remote = await loadRemoteVersion(this.fetchImpl)
      const update = determineUpdate(localStatus, remote)
      if (!update) return null
      const release = update.type === 'release'
      const label = release
        ? t('updates.releaseAvailable', {}, 'New version available')
        : t('updates.snapshotAvailable', {}, 'New snapshot build')
      const title = release
        ? t('updates.releaseTitle', {}, 'Open the latest VoxCPMTTS release')
        : t('updates.snapshotTitle', {}, 'View the newer snapshot source')
      this.element.href = update.url
      this.element.title = title
      this.element.setAttribute('aria-label', title)
      this.element.querySelector('span').textContent = label
      this.element.dataset.updateType = update.type
      this.element.hidden = false
      return update
    } catch {
      return null
    }
  }
}
