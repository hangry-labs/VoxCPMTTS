import { AudioEditor } from './audio-editor.js'
import { AudioRecorder } from './audio-recorder.js?v=voice-library'

const $ = (selector, root = document) => root.querySelector(selector)
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)]

const UI_SESSION_KEY = 'voxcpmtts-ui-state-v1'
const GPU_SESSION_KEY = 'voxcpmtts-gpu-history-v1'
const GPU_HISTORY_RETENTION_MS = 10 * 60 * 1000
const GPU_POLL_INTERVAL_MS = 1000
const SAMPLE_TEXTS = [
  'VoxCPM2 generates natural multilingual speech with voice design and cloning.',
  'A steady voice can make technical information easier to understand.',
  'Good morning. The latest build is ready for a careful listening test.',
  'This reference voice can speak new text while preserving its character and pacing.',
]
const INPUT_SAMPLES = {
  ssml: [
    `<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  Welcome to VoxCPMTTS.<break time="350ms"/>
  <prosody rate="90%" pitch="+1st">This line is slower and slightly brighter.</prosody>
</speak>`,
  ],
  'ssml-h': [
    `<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
  xmlns:h="https://hangrylabs.app/ns/ssml-h/1.0" xml:lang="en-US">
  <metadata><h:extensions version="1.0">
    <h:voice-definition name="Host" gender="female" style="warm and confident">
      <h:sample xml:lang="en-US">Welcome. I will guide our conversation today.</h:sample>
    </h:voice-definition>
    <h:voice-definition name="Guest">
      <h:description>A thoughtful male guest with a relaxed, conversational delivery.</h:description>
    </h:voice-definition>
  </h:extensions></metadata>
  <voice name="Host">Welcome to the show. What are we exploring today?</voice>
  <voice name="Guest">We are testing a complete multi-speaker discussion from one document.</voice>
</speak>`,
  ],
}
const AUDIO_EDITOR_LABELS = {
  noAudio: 'No audio selected',
  download: 'Download audio',
  share: 'Share audio',
  remove: 'Remove audio',
  mute: 'Mute or unmute',
  volume: 'Volume',
  playbackSpeed: 'Playback speed',
  backward: 'Seek backward 5 seconds',
  play: 'Play',
  pause: 'Pause',
  forward: 'Seek forward 5 seconds',
  restart: 'Return to start',
  trim: 'Select and trim audio',
  cancel: 'Cancel',
  applySelection: 'Apply selection',
}
const RECORDER_LABELS = {
  record: 'Record',
  stop: 'Stop recording',
  ready: 'Ready to record',
  recording: 'Recording {time}',
  processing: 'Preparing recording',
  readyWithTime: 'Recording ready · {time}',
  unavailable: 'Microphone recording requires a supported browser and secure connection.',
}
const GPU_METRICS = [
  { key: 'utilization', label: 'GPU utilization', color: '#ff7a1a' },
  { key: 'memory_utilization', label: 'Memory activity', color: '#c586c0' },
  { key: 'memory_used', label: 'VRAM', color: '#72a7ff' },
  { key: 'temperature', label: 'Temperature', color: '#ef6b73' },
  { key: 'power', label: 'Power', color: '#f2c94c' },
  { key: 'fan_speed', label: 'Fan', color: '#55c58a' },
  { key: 'graphics_clock', label: 'Graphics clock', color: '#9cdcfe' },
  { key: 'memory_clock', label: 'Memory clock', color: '#ce9178' },
]

const state = {
  activeTab: 'generate',
  inputType: 'text',
  inputDrafts: { text: null, ssml: null, 'ssml-h': null },
  headerCollapsed: document.documentElement.dataset.headerCollapsed === 'true',
  streamAbort: null,
  streamPlayback: null,
  sampleIndex: 0,
  gpuHistory: new Map(),
  gpuStats: [],
  gpuWindowMs: 60 * 1000,
  gpuTimer: null,
  gpuRefreshActive: false,
  gpuHovering: false,
  formats: {},
  streamFormats: {},
  defaults: {},
  status: {},
  profiles: [],
  quickSaveType: null,
  editingProfile: null,
  pendingDeleteProfile: null,
  activityTimer: null,
}

const generateOutput = new AudioEditor($('#generate-output'), {
  label: 'Generated audio',
  emptyTitle: 'Audio output',
  emptyDescription: 'Ready for synthesis',
  labels: AUDIO_EDITOR_LABELS,
})
const streamOutput = new AudioEditor($('#stream-output'), {
  label: 'Streamed audio',
  emptyTitle: 'Audio output',
  emptyDescription: 'Ready for streaming',
  labels: AUDIO_EDITOR_LABELS,
})
const cloneOutput = new AudioEditor($('#clone-output'), {
  label: 'Cloned audio',
  emptyTitle: 'Audio output',
  emptyDescription: 'Ready for voice cloning',
  labels: AUDIO_EDITOR_LABELS,
})
const referenceAudio = new AudioEditor($('#reference-audio-preview'), {
  label: 'Reference preview',
  emptyTitle: 'No reference selected',
  emptyDescription: 'Choose or record a sample',
  labels: AUDIO_EDITOR_LABELS,
  onChange: (file) => {
    $('#reference-audio-drop').classList.toggle('has-file', Boolean(file))
    $('#reference-audio-name').textContent = file?.name || 'WAV, MP3, FLAC, OGG, or M4A'
    updateWorkflowControls()
  },
})
const profileAudio = new AudioEditor($('#profile-audio-preview'), {
  label: 'Voice sample',
  emptyTitle: 'No voice sample selected',
  emptyDescription: 'Choose or drop a clean reference recording',
  labels: AUDIO_EDITOR_LABELS,
  onChange: (file) => {
    $('#profile-audio-drop').classList.toggle('has-file', Boolean(file))
    $('#profile-audio-name').textContent = file?.name || 'WAV, MP3, FLAC, OGG, or M4A'
  },
})
const referenceRecorder = new AudioRecorder({
  button: $('#reference-record-toggle'),
  status: $('#reference-record-state'),
  canvas: $('#reference-record-wave'),
  labels: RECORDER_LABELS,
  onFile: (file) => referenceAudio.load(file, file.name),
  onError: (error) => showToast(errorMessage(error)),
})

for (const editor of [generateOutput, cloneOutput, streamOutput, referenceAudio, profileAudio]) {
  editor.container.addEventListener('audio-error', (event) => showToast(errorMessage(event.detail)))
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

async function responseError(response) {
  const text = await response.text()
  try {
    const payload = JSON.parse(text)
    return payload.detail || payload.error?.message || text
  } catch {
    return text || `HTTP ${response.status}`
  }
}

async function fetchJson(path, options) {
  const response = await fetch(path, options)
  if (!response.ok) throw new Error(await responseError(response))
  return response.json()
}

function showToast(message, tone = 'error') {
  const toast = $('#toast')
  toast.textContent = message
  toast.dataset.tone = tone
  toast.hidden = false
  clearTimeout(showToast.timer)
  showToast.timer = setTimeout(() => { toast.hidden = true }, 5000)
}

function setStatus(message, tone = 'neutral') {
  const status = $('#global-status')
  status.textContent = message
  status.dataset.tone = tone
}

function persistUiState() {
  try {
    sessionStorage.setItem(UI_SESSION_KEY, JSON.stringify({
      activeTab: state.activeTab,
      headerCollapsed: state.headerCollapsed,
      gpuWindowMs: state.gpuWindowMs,
      inputType: state.inputType,
    }))
  } catch {
    // Browser storage may be unavailable in privacy-restricted sessions.
  }
}

function readSessionJson(key) {
  try { return JSON.parse(sessionStorage.getItem(key) || 'null') } catch { return null }
}

function persistGpuSession() {
  try {
    sessionStorage.setItem(GPU_SESSION_KEY, JSON.stringify({
      savedAt: Date.now(),
      stats: state.gpuStats,
      history: Object.fromEntries(state.gpuHistory),
    }))
  } catch {
    // Monitoring continues in memory when session storage is unavailable.
  }
}

function restoreSessionState() {
  const ui = readSessionJson(UI_SESSION_KEY)
  if (['generate', 'clone', 'stream', 'voices', 'api', 'system'].includes(ui?.activeTab)) {
    state.activeTab = ui.activeTab
  }
  if (typeof ui?.headerCollapsed === 'boolean') state.headerCollapsed = ui.headerCollapsed
  if ([60 * 1000, 10 * 60 * 1000].includes(ui?.gpuWindowMs)) state.gpuWindowMs = ui.gpuWindowMs
  if (['text', 'ssml', 'ssml-h'].includes(ui?.inputType)) state.inputType = ui.inputType

  const cached = readSessionJson(GPU_SESSION_KEY)
  const cutoff = Date.now() - GPU_HISTORY_RETENTION_MS
  if (!cached || !Number.isFinite(cached.savedAt) || cached.savedAt < cutoff) return
  if (Array.isArray(cached.stats)) state.gpuStats = cached.stats
  Object.entries(cached.history || {}).forEach(([index, samples]) => {
    const recent = Array.isArray(samples)
      ? samples.filter((sample) => Number.isFinite(sample?.timestamp) && sample.timestamp >= cutoff)
      : []
    if (recent.length) state.gpuHistory.set(Number(index), recent)
  })
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

function updateMetrics() {
  const text = $('#text-input').value
  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  $('#text-metrics').textContent = `${text.length} characters · ${words} words`
}

function inputSample(inputType) {
  if (inputType === 'text') return SAMPLE_TEXTS[state.sampleIndex]
  return INPUT_SAMPLES[inputType][0]
}

function setInputType(inputType) {
  if (!['text', 'ssml', 'ssml-h'].includes(inputType)) return
  const editor = $('#text-input')
  const currentType = editor.dataset.inputType || state.inputType
  state.inputDrafts[currentType] = editor.value
  if (state.inputDrafts[inputType] == null) state.inputDrafts[inputType] = inputSample(inputType)
  editor.value = state.inputDrafts[inputType]
  editor.dataset.inputType = inputType
  editor.spellcheck = inputType === 'text'
  state.inputType = inputType
  $$('.input-type-control button').forEach((button) => {
    const active = button.dataset.inputType === inputType
    button.classList.toggle('active', active)
    button.setAttribute('aria-pressed', String(active))
  })
  updateMetrics()
  persistUiState()
}

function populateSelect(select, options, selected) {
  select.replaceChildren(...options.map(({ value, label }) => {
    const option = document.createElement('option')
    option.value = value
    option.textContent = label
    return option
  }))
  if (options.some((option) => option.value === selected)) select.value = selected
}

function normalizedProfileName(value) {
  return value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^[-_]+|[-_]+$/g, '').slice(0, 48)
}

function selectedProfile() {
  const id = $('#voice-profile').value
  return state.profiles.find((profile) => profile.id === id) || null
}

function updateVoiceProfileState() {
  const profile = selectedProfile()
  const note = $('#voice-profile-note')
  if (!profile) note.textContent = 'Use a saved design or clone sample'
  else if (profile.profile_type === 'cloned') note.textContent = `${profile.id} · stored reference audio${profile.has_transcript ? ' · transcript' : ''}`
  else note.textContent = `${profile.id} · saved voice design`
  updateWorkflowControls()
}

function renderVoiceProfileSelect(selected = $('#voice-profile').value) {
  populateSelect($('#voice-profile'), [
    { value: '', label: 'None' },
    ...state.profiles.map((profile) => ({
      value: profile.id,
      label: `${profile.id} (${profile.profile_type === 'cloned' ? 'clone' : 'design'})`,
    })),
  ], selected)
  updateVoiceProfileState()
}

function useProfile(profile, tab) {
  renderVoiceProfileSelect(profile.id)
  if (profile.language) $('#language').value = profile.language
  activateTab(tab)
  setStatus(`Voice ${profile.id} selected`, 'success')
}

function renderProfileList() {
  const list = $('#profile-list')
  const query = $('#profile-filter').value.trim().toLowerCase()
  const profiles = state.profiles.filter((profile) => `${profile.id} ${profile.description} ${profile.profile_type}`.toLowerCase().includes(query))
  if (!profiles.length) {
    const empty = document.createElement('div')
    empty.className = 'empty-profile-list'
    empty.textContent = state.profiles.length ? 'No saved voices match this search.' : 'No saved voices yet.'
    list.replaceChildren(empty)
    return
  }
  list.replaceChildren(...profiles.map((profile) => {
    const card = document.createElement('article')
    card.className = 'profile-card'
    const copy = document.createElement('div')
    const title = document.createElement('strong')
    title.textContent = profile.id
    const description = document.createElement('p')
    description.className = 'profile-description'
    description.textContent = profile.description || (profile.profile_type === 'cloned' ? 'Saved reference voice' : 'Saved voice design')
    const metadata = document.createElement('div')
    metadata.className = 'profile-metadata'
    ;[
      profile.profile_type === 'cloned' ? 'Cloned' : 'Designed',
      profile.language || 'Automatic language',
      profile.has_transcript ? 'Transcript saved' : null,
    ].filter(Boolean).forEach((label) => {
      const badge = document.createElement('span')
      badge.className = 'profile-badge'
      badge.textContent = label
      metadata.append(badge)
    })
    copy.append(title, description, metadata)
    if (profile.audio_url) {
      const audio = document.createElement('audio')
      audio.className = 'profile-audio'
      audio.controls = true
      audio.preload = 'none'
      audio.src = profile.audio_url
      copy.append(audio)
    }
    const actions = document.createElement('div')
    actions.className = 'profile-actions'
    const useGenerate = document.createElement('button')
    useGenerate.type = 'button'
    useGenerate.className = 'secondary-button profile-use'
    useGenerate.innerHTML = '<i class="icon-audio-lines"></i><span>Generate</span>'
    useGenerate.addEventListener('click', () => useProfile(profile, 'generate'))
    actions.append(useGenerate)
    if (profile.profile_type === 'cloned') {
      const useClone = document.createElement('button')
      useClone.type = 'button'
      useClone.className = 'secondary-button profile-use'
      useClone.innerHTML = '<i class="icon-mic"></i><span>Clone</span>'
      useClone.addEventListener('click', () => useProfile(profile, 'clone'))
      actions.append(useClone)
    }
    const remove = document.createElement('button')
    remove.type = 'button'
    remove.className = 'icon-button bordered danger-icon'
    remove.title = `Delete ${profile.id}`
    remove.setAttribute('aria-label', `Delete ${profile.id}`)
    remove.innerHTML = '<i class="icon-x"></i>'
    remove.addEventListener('click', () => openDeleteProfileDialog(profile))
    if (profile.profile_type === 'cloned') {
      const edit = document.createElement('button')
      edit.type = 'button'
      edit.className = 'secondary-button profile-use'
      edit.title = `Edit ${profile.id}`
      edit.setAttribute('aria-label', `Edit ${profile.id}`)
      edit.innerHTML = '<i class="icon-sliders-horizontal"></i><span>Edit</span>'
      edit.addEventListener('click', () => openProfileEditor(profile))
      actions.append(edit)
    }
    actions.append(remove)
    card.append(copy, actions)
    return card
  }))
}

async function refreshProfiles(selected) {
  const payload = await fetchJson('/tts/voice-profiles')
  state.profiles = payload.data || []
  $('#profile-count').textContent = String(state.profiles.length)
  renderVoiceProfileSelect(selected)
  renderProfileList()
}

function formatOptions(formats) {
  return Object.entries(formats).map(([value, details]) => ({
    value,
    label: details.label || value.toUpperCase(),
  }))
}

function refreshFormatOptions() {
  const streaming = state.activeTab === 'stream'
  const formats = streaming ? state.streamFormats : state.formats
  const preferred = $('#output-format').value || (streaming ? 'mp3' : 'mp3')
  populateSelect($('#output-format'), formatOptions(formats), preferred)
}

function updateVoiceDesignState() {
  const active = Boolean($('#control-input').value.trim())
  const indicator = $('#voice-design-state')
  indicator.textContent = active ? 'Active' : 'Optional'
  indicator.classList.toggle('active', active)
  $('#save-designed-voice').disabled = !active
}

function updateCloneConditioning() {
  const guided = Boolean($('#reference-text').value.trim())
  const direction = $('#clone-control-input')
  direction.disabled = guided
  $('#clone-direction-state').textContent = guided ? 'Transcript guided' : 'Optional'
}

function updateSeedState() {
  $('#seed').disabled = $('#randomize-seed').checked
}

function updateTimestampState() {
  const enabled = $('#generate-timestamps')
  $('#timestamp-level').disabled = enabled.disabled || !enabled.checked
}

function updateWorkflowControls() {
  const cloning = state.activeTab === 'clone'
  const profile = selectedProfile()
  $('#voice-design-details').hidden = cloning
  $('#reference-audio').disabled = !cloning
  $('#denoise').disabled = !cloning || !state.status.load_denoiser
  $('#save-cloned-voice').disabled = !referenceAudio.currentFile()
  $('#voice-profile-note').classList.toggle('active', Boolean(profile))
}

function buildPayload({ workflow = state.activeTab, streaming = false } = {}) {
  const cloning = workflow === 'clone'
  const outputFormat = streaming ? 'mp3' : $('#output-format').value
  const referenceText = cloning ? ($('#reference-text').value.trim() || null) : null
  const payload = {
    text: $('#text-input').value.trim(),
    input_type: state.inputType,
    language: $('#language').value || 'English',
    voice: cloning ? 'reference' : 'auto',
    voice_profile: $('#voice-profile').value || null,
    control: cloning
      ? (referenceText ? null : ($('#clone-control-input').value.trim() || null))
      : ($('#control-input').value.trim() || null),
    ref_text: referenceText,
    cfg_value: Number($('#guidance').value),
    inference_timesteps: Number($('#steps').value),
    normalize: $('#normalize-text').checked,
    normalize_loudness: $('#normalize-loudness').checked,
    denoise: cloning && $('#denoise').checked,
    seed: Number($('#seed').value || 42),
    randomize_seed: $('#randomize-seed').checked,
    device: $('#device').value,
    output_format: outputFormat,
  }
  if (streaming) payload.stream_format = 'mp3'
  return payload
}

async function requestAudioResponse({ workflow = state.activeTab, streaming = false, signal } = {}) {
  const payload = buildPayload({ workflow, streaming })
  if (!payload.text) throw new Error('Enter text to synthesize.')
  const needsReference = workflow === 'clone'
  const reference = referenceAudio.currentFile()
  const profile = selectedProfile()
  if (needsReference && !reference && profile?.profile_type !== 'cloned') {
    throw new Error('Choose or record reference audio, or select a saved cloned voice.')
  }

  const route = streaming ? '/tts/stream' : '/tts/generate'
  let response
  if (needsReference && reference) {
    const form = new FormData()
    form.append('payload', JSON.stringify(payload))
    form.append('reference_audio', reference, reference.name)
    response = await fetch(`${route}-upload`, { method: 'POST', body: form, signal })
  } else {
    response = await fetch(route, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    })
  }
  if (!response.ok) throw new Error(await responseError(response))
  return { response, payload }
}

async function requestAudio(options = {}) {
  const { response, payload } = await requestAudioResponse(options)
  return {
    blob: await response.blob(),
    extension: response.headers.get('X-VoxCPM-Format') || payload.output_format,
    seed: response.headers.get('X-VoxCPM-Seed'),
    payload,
  }
}

function renderTimestamps(workflow, result) {
  const panel = $(`#${workflow}-timestamp-results`)
  const header = document.createElement('header')
  const title = document.createElement('strong')
  title.textContent = `${result.level[0].toUpperCase()}${result.level.slice(1)} timestamps`
  const count = document.createElement('span')
  count.textContent = `${result.items.length} aligned`
  header.append(title, count)
  const list = document.createElement('ol')
  list.className = 'timestamp-list'
  result.items.forEach((item) => {
    const row = document.createElement('li')
    const start = document.createElement('span')
    const end = document.createElement('span')
    const text = document.createElement('span')
    start.textContent = `${Number(item.start).toFixed(2)}s`
    end.textContent = `${Number(item.end).toFixed(2)}s`
    text.textContent = item.text
    row.append(start, end, text)
    list.append(row)
  })
  panel.replaceChildren(header, list)
  panel.hidden = false
}

async function alignGeneratedAudio(workflow, blob, extension, payload) {
  const panel = $(`#${workflow}-timestamp-results`)
  if (!$('#generate-timestamps').checked) {
    panel.hidden = true
    return
  }
  setStatus('Aligning generated speech')
  const form = new FormData()
  form.append('audio', blob, `voxcpmtts.${extension}`)
  form.append('text', payload.text)
  form.append('level', $('#timestamp-level').value)
  form.append('language', payload.language || '')
  try {
    const result = await fetchJson('/tts/timestamps-upload', { method: 'POST', body: form })
    renderTimestamps(workflow, result)
  } catch (error) {
    panel.hidden = true
    showToast(`Audio is ready, but timestamp alignment failed: ${errorMessage(error)}`)
  }
}

const ACTIVITY_STAGE_INDEX = {
  preparing: 0,
  loading_model: 0,
  generating: 1,
  streaming: 1,
  encoding: 2,
  complete: 3,
  failed: -1,
}

function renderInferenceProgress(workflow, activity) {
  if (workflow === 'stream') return
  const panel = $(`#${workflow}-progress`)
  const phase = activity.phase || 'preparing'
  const index = ACTIVITY_STAGE_INDEX[phase] ?? 0
  panel.hidden = false
  panel.dataset.phase = phase
  $('.progress-copy strong', panel).textContent = activity.message || 'Preparing request'
  $('.progress-copy > span', panel).textContent = phase === 'loading_model'
    ? 'First use can take longer while model weights enter GPU memory'
    : phase === 'encoding' ? 'Preparing the selected output format' : 'The request is active'
  $$('[data-stage]', panel).forEach((dot, dotIndex) => {
    dot.classList.toggle('done', phase === 'complete' || dotIndex < index)
    dot.classList.toggle('active', phase !== 'complete' && phase !== 'failed' && dotIndex === index)
  })
}

function startActivityPolling(workflow) {
  clearInterval(state.activityTimer)
  renderInferenceProgress(workflow, { phase: 'preparing', message: 'Preparing request' })
  state.activityTimer = setInterval(async () => {
    try {
      renderInferenceProgress(workflow, await fetchJson('/tts/activity', { cache: 'no-store' }))
    } catch {
      // The generation request remains authoritative if progress polling is interrupted.
    }
  }, 300)
}

function finishActivityPolling(workflow, phase, message) {
  clearInterval(state.activityTimer)
  state.activityTimer = null
  if (workflow === 'stream') return
  renderInferenceProgress(workflow, { phase, message })
  const panel = $(`#${workflow}-progress`)
  clearTimeout(panel.hideTimer)
  panel.hideTimer = setTimeout(() => { panel.hidden = true }, phase === 'complete' ? 1800 : 5000)
}

function setGenerationBusy(active, workflow = 'generate') {
  const button = workflow === 'stream' ? $('#stream-start') : $(`#${workflow}-button`)
  button.disabled = active
  if (workflow === 'stream') {
    $('#stream-stop').disabled = !active
    $('#stream-progress').hidden = !active
  }
}

async function generateAudio(workflow = 'generate') {
  const output = workflow === 'clone' ? cloneOutput : generateOutput
  setGenerationBusy(true, workflow)
  startActivityPolling(workflow)
  setStatus('Generating audio')
  try {
    const { blob, extension, seed, payload } = await requestAudio({ workflow })
    await output.load(blob, `voxcpmtts${workflow === 'clone' ? '-clone' : ''}.${extension}`)
    if (seed !== null) {
      $('#last-generated-seed').value = seed
      $('#seed').value = seed
    }
    await alignGeneratedAudio(workflow, blob, extension, payload)
    setStatus('Generation complete', 'success')
    finishActivityPolling(workflow, 'complete', 'Audio is ready')
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    showToast(errorMessage(error))
    finishActivityPolling(workflow, 'failed', errorMessage(error))
  } finally {
    setGenerationBusy(false, workflow)
  }
}

class StreamWaveform {
  constructor(canvas, surface, stateElement, detailElement) {
    this.canvas = canvas
    this.surface = surface
    this.stateElement = stateElement
    this.detailElement = detailElement
    this.peaks = []
    this.animationFrame = null
    this.totalBytes = 0
    this.drawIdle()
  }

  async attach(audio) {
    this.stopAudioGraph()
    this.surface.hidden = false
    this.stateElement.textContent = 'Waiting for first audio chunk'
    this.detailElement.textContent = '0 KiB buffered'
    this.peaks = []
    this.audioContext = new AudioContext()
    this.source = this.audioContext.createMediaElementSource(audio)
    this.analyser = this.audioContext.createAnalyser()
    this.analyser.fftSize = 1024
    this.source.connect(this.analyser)
    this.analyser.connect(this.audioContext.destination)
    await this.audioContext.resume()
    this.draw()
  }

  update(totalBytes, currentTime = 0) {
    this.totalBytes = totalBytes
    this.stateElement.textContent = totalBytes ? 'Playing generated speech' : 'Waiting for first audio chunk'
    this.detailElement.textContent = `${(totalBytes / 1024).toFixed(0)} KiB buffered · ${formatClock(currentTime)}`
  }

  draw() {
    if (!this.analyser) return
    const samples = new Uint8Array(this.analyser.fftSize)
    this.analyser.getByteTimeDomainData(samples)
    const peak = samples.reduce((maximum, sample) => Math.max(maximum, Math.abs(sample - 128) / 128), 0)
    this.peaks.push(Math.max(0.025, peak))
    const maximumPeaks = Math.max(80, Math.round(this.canvas.clientWidth / 3))
    if (this.peaks.length > maximumPeaks) this.peaks.splice(0, this.peaks.length - maximumPeaks)
    this.drawPeaks()
    this.animationFrame = requestAnimationFrame(() => this.draw())
  }

  drawPeaks() {
    const context = this.canvas.getContext('2d')
    const ratio = window.devicePixelRatio || 1
    const width = this.canvas.width = Math.max(1, Math.round(this.canvas.clientWidth * ratio))
    const height = this.canvas.height = Math.max(1, Math.round(this.canvas.clientHeight * ratio))
    context.clearRect(0, 0, width, height)
    context.strokeStyle = '#2b2d33'
    context.lineWidth = ratio
    context.beginPath()
    context.moveTo(0, height / 2)
    context.lineTo(width, height / 2)
    context.stroke()
    if (!this.peaks.length) return
    const spacing = width / Math.max(1, this.peaks.length)
    context.strokeStyle = '#ff7a1a'
    context.lineWidth = Math.max(ratio, spacing * 0.48)
    context.beginPath()
    this.peaks.forEach((value, index) => {
      const x = (index + 0.5) * spacing
      const amplitude = Math.max(2 * ratio, value * height * 0.44)
      context.moveTo(x, height / 2 - amplitude)
      context.lineTo(x, height / 2 + amplitude)
    })
    context.stroke()
  }

  complete() {
    this.stateElement.textContent = 'Stream complete'
    this.detailElement.textContent = `${(this.totalBytes / 1024).toFixed(0)} KiB received`
  }

  hide() {
    this.surface.hidden = true
    this.stopAudioGraph()
  }

  stopAudioGraph() {
    cancelAnimationFrame(this.animationFrame)
    this.source?.disconnect()
    this.analyser?.disconnect()
    if (this.audioContext?.state !== 'closed') this.audioContext?.close().catch(() => {})
    this.audioContext = null
    this.source = null
    this.analyser = null
    this.animationFrame = null
  }

  drawIdle() {
    this.peaks = []
    this.drawPeaks()
  }
}

function formatClock(seconds) {
  const safe = Math.max(0, Number(seconds) || 0)
  return `${Math.floor(safe / 60)}:${Math.floor(safe % 60).toString().padStart(2, '0')}`
}

const streamWaveform = new StreamWaveform($('#stream-live-wave'), $('#stream-live'), $('#stream-live-state'), $('#stream-live-detail'))

class IncrementalAudioPlayback {
  static async create(visualizer) {
    if (!window.MediaSource || !MediaSource.isTypeSupported('audio/mpeg')) return null
    const mediaSource = new MediaSource()
    const objectUrl = URL.createObjectURL(mediaSource)
    const audio = new Audio(objectUrl)
    await new Promise((resolve, reject) => {
      mediaSource.addEventListener('sourceopen', resolve, { once: true })
      mediaSource.addEventListener('error', reject, { once: true })
    })
    try {
      const playback = new IncrementalAudioPlayback(mediaSource, audio, objectUrl)
      await visualizer.attach(audio)
      return playback
    } catch {
      URL.revokeObjectURL(objectUrl)
      return null
    }
  }

  constructor(mediaSource, audio, objectUrl) {
    this.mediaSource = mediaSource
    this.audio = audio
    this.objectUrl = objectUrl
    this.sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg')
    this.queue = Promise.resolve()
    this.started = false
  }

  append(chunk) {
    const bytes = chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength)
    this.queue = this.queue.then(() => new Promise((resolve, reject) => {
      const done = () => { this.sourceBuffer.removeEventListener('error', failed); resolve() }
      const failed = () => { this.sourceBuffer.removeEventListener('updateend', done); reject(new Error('Browser could not buffer streamed MP3 audio.')) }
      this.sourceBuffer.addEventListener('updateend', done, { once: true })
      this.sourceBuffer.addEventListener('error', failed, { once: true })
      this.sourceBuffer.appendBuffer(bytes)
    })).then(() => {
      if (!this.started) { this.started = true; this.audio.play().catch(() => {}) }
    })
    return this.queue
  }

  async finish() {
    await this.queue
    if (this.mediaSource.readyState === 'open') this.mediaSource.endOfStream()
  }

  currentTime() { return this.audio.currentTime || 0 }

  stop() {
    this.audio.pause()
    if (this.mediaSource.readyState === 'open') {
      try { this.mediaSource.endOfStream() } catch {}
    }
    URL.revokeObjectURL(this.objectUrl)
  }
}

async function streamAudio() {
  const controller = new AbortController()
  const chunks = []
  let playback = null
  state.streamAbort = controller
  setGenerationBusy(true, 'stream')
  streamWaveform.surface.hidden = false
  streamWaveform.stateElement.textContent = 'Preparing stream'
  streamWaveform.detailElement.textContent = '0 KiB buffered'
  setStatus('Streaming audio')
  try {
    playback = await IncrementalAudioPlayback.create(streamWaveform)
    state.streamPlayback = playback
    const { response } = await requestAudioResponse({ workflow: 'stream', streaming: true, signal: controller.signal })
    const seed = response.headers.get('X-VoxCPM-Seed')
    if (seed !== null) {
      $('#last-generated-seed').value = seed
      $('#seed').value = seed
    }
    if (!response.body) throw new Error('Streaming response body is unavailable in this browser.')
    const reader = response.body.getReader()
    let totalBytes = 0
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      totalBytes += value.byteLength
      if (playback) {
        try {
          await playback.append(value)
        } catch (error) {
          playback.stop()
          playback = null
          state.streamPlayback = null
          showToast(errorMessage(error))
        }
      }
      streamWaveform.update(totalBytes, playback?.currentTime() || 0)
      setStatus(`Streaming audio · ${(totalBytes / 1024).toFixed(0)} KiB`)
    }
    let resumeAt = 0
    if (playback) {
      await playback.finish()
      resumeAt = playback.currentTime()
      playback.stop()
      playback = null
      state.streamPlayback = null
    }
    await streamOutput.load(new Blob(chunks, { type: 'audio/mpeg' }), 'voxcpmtts-stream.mp3')
    if (resumeAt > 0) await streamOutput.playFrom(resumeAt).catch(() => {})
    streamWaveform.complete()
    setTimeout(() => streamWaveform.hide(), 1000)
    setStatus('Stream complete', 'success')
  } catch (error) {
    if (error.name === 'AbortError') {
      if (chunks.length) await streamOutput.load(new Blob(chunks, { type: 'audio/mpeg' }), 'voxcpmtts-stream-partial.mp3')
      streamWaveform.complete()
      setStatus('Stream stopped', 'success')
    }
    else {
      setStatus(errorMessage(error), 'error')
      showToast(errorMessage(error))
    }
  } finally {
    playback?.stop()
    state.streamAbort = null
    state.streamPlayback = null
    setGenerationBusy(false, 'stream')
  }
}

async function transcribeReference() {
  const file = referenceAudio.currentFile()
  if (!file) return showToast('Choose or record reference audio first.')
  const button = $('#transcribe-reference')
  const label = button.querySelector('span')
  button.disabled = true
  const firstLoad = !state.status.asr_loaded
  label.textContent = firstLoad ? 'Loading transcription model' : 'Transcribing reference'
  setStatus(firstLoad ? 'Loading transcription model' : 'Transcribing reference')
  try {
    const form = new FormData()
    form.append('reference_audio', file, file.name)
    form.append('language', 'auto')
    const result = await fetchJson('/tts/transcribe-upload', { method: 'POST', body: form })
    $('#reference-text').value = result.text || ''
    state.status.asr_loaded = true
    updateCloneConditioning()
    setStatus('Reference transcript ready', 'success')
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    showToast(errorMessage(error))
  } finally {
    label.textContent = 'Transcribe reference'
    button.disabled = !state.status.load_asr
  }
}

async function saveProfileRequest({ name, profileType, description = '', file = null, refText = '', control = '', editing = false }) {
  const form = new FormData()
  form.append('name', name)
  form.append('profile_type', profileType)
  form.append('description', description)
  form.append('ref_text', refText)
  form.append('control', control)
  form.append('language', $('#language').value || 'English')
  if (file) form.append('reference_audio', file, file.name)
  const path = editing ? `/tts/voice-profiles/${encodeURIComponent(name)}` : '/tts/voice-profiles'
  return fetchJson(path, { method: editing ? 'PUT' : 'POST', body: form })
}

function openQuickSaveDialog(profileType) {
  if (profileType === 'designed' && !$('#control-input').value.trim()) {
    return showToast('Enter a voice description before saving this design.')
  }
  if (profileType === 'cloned' && !referenceAudio.currentFile()) {
    return showToast('Choose or record reference audio before saving this voice.')
  }
  state.quickSaveType = profileType
  $('#save-profile-title').textContent = profileType === 'designed' ? 'Save designed voice' : 'Save reference voice'
  $('#save-profile-copy').textContent = profileType === 'designed'
    ? 'Reuse this voice description from Generate or Stream.'
    : 'Store this sample for one-click cloning from any generation screen.'
  $('#quick-profile-name').value = ''
  $('#quick-profile-description').value = ''
  $('#save-profile-dialog').showModal()
  $('#quick-profile-name').focus()
}

function closeQuickSaveDialog() {
  state.quickSaveType = null
  $('#save-profile-dialog').close()
}

function openDeleteProfileDialog(profile) {
  state.pendingDeleteProfile = profile
  $('#delete-profile-name').textContent = profile.id
  $('#delete-profile-dialog').showModal()
}

function closeDeleteProfileDialog() {
  state.pendingDeleteProfile = null
  $('#delete-profile-dialog').close()
}

function resetProfileEditor() {
  state.editingProfile = null
  $('#voice-form').reset()
  $('#profile-name').disabled = false
  $('#voice-form-title').textContent = 'Add a reference voice'
  $('#voice-form-copy').textContent = 'Drop a clean voice sample here, then use it from Generate, Clone, or Stream.'
  $('#profile-submit-label').textContent = 'Save voice'
  $('#profile-edit-cancel').hidden = true
  profileAudio.clear()
  updateProfileNamePreview()
}

function openProfileEditor(profile) {
  state.editingProfile = profile
  $('#profile-name').value = profile.id
  $('#profile-name').disabled = true
  $('#profile-description').value = profile.description || ''
  $('#profile-transcript').value = profile.ref_text || ''
  if (profile.language) $('#language').value = profile.language
  $('#voice-form-title').textContent = `Edit ${profile.id}`
  $('#voice-form-copy').textContent = 'Update its details or choose a new sample. The current sample is retained when none is selected.'
  $('#profile-submit-label').textContent = 'Update voice'
  $('#profile-edit-cancel').hidden = false
  profileAudio.clear()
  updateProfileNamePreview()
  activateTab('voices')
  $('#profile-description').focus()
}

async function loadProfileAudio(file) {
  if (!file) return
  try {
    await profileAudio.load(file, file.name)
    if (!$('#profile-name').value.trim()) $('#profile-name').value = normalizedProfileName(file.name.replace(/\.[^.]+$/, ''))
    updateProfileNamePreview()
  } catch (error) {
    profileAudio.clear()
    showToast(errorMessage(error))
  }
}

function updateProfileNamePreview() {
  const normalized = normalizedProfileName($('#profile-name').value)
  const preview = $('#profile-name-preview')
  if (state.editingProfile) preview.textContent = `Editing ${state.editingProfile.id}`
  else if (!normalized) preview.textContent = 'Letters, numbers, hyphens, and underscores'
  else if (state.profiles.some((profile) => profile.id === normalized)) preview.textContent = `${normalized} already exists`
  else preview.textContent = `Saved as ${normalized}`
}

async function refreshApi() {
  const [status, defaults, formats, languages, voices] = await Promise.all([
    fetchJson('/tts/status'),
    fetchJson('/tts/defaults'),
    fetchJson('/tts/formats'),
    fetchJson('/tts/languages'),
    fetchJson('/tts/voices'),
  ])
  $('#api-output').textContent = JSON.stringify(status, null, 2)
  $('#capabilities-output').textContent = JSON.stringify({ defaults, formats, languages, voices }, null, 2)
}

function element(tag, className, text) {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function gpuHistoryPoints(samples, now, windowMs, metricKey, maximum, width = 300, height = 70) {
  const windowStart = now - windowMs
  return samples.filter((sample) => Number.isFinite(sample[metricKey])).map((sample) => {
    const x = Math.min(width, Math.max(0, (sample.timestamp - windowStart) / windowMs * width))
    const y = height - (Math.min(maximum, Math.max(0, sample[metricKey])) / maximum * height)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function addGpuChartGrid(svg, width, height) {
  for (let column = 0; column <= 10; column += 1) {
    const x = column * width / 10
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.setAttribute('class', 'gpu-grid-line')
    line.setAttribute('x1', String(x))
    line.setAttribute('x2', String(x))
    line.setAttribute('y1', '0')
    line.setAttribute('y2', String(height))
    svg.append(line)
  }
  for (let row = 0; row <= 4; row += 1) {
    const y = row * height / 4
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.setAttribute('class', 'gpu-grid-line')
    line.setAttribute('x1', '0')
    line.setAttribute('x2', String(width))
    line.setAttribute('y1', String(y))
    line.setAttribute('y2', String(y))
    svg.append(line)
  }
}

function mergeGpuHistory(historyPayload) {
  const cutoff = Date.now() - GPU_HISTORY_RETENTION_MS
  Object.entries(historyPayload || {}).forEach(([index, incoming]) => {
    if (!Array.isArray(incoming)) return
    const byTimestamp = new Map()
    ;[...(state.gpuHistory.get(Number(index)) || []), ...incoming].forEach((sample) => {
      if (Number.isFinite(sample?.timestamp) && sample.timestamp >= cutoff) byTimestamp.set(sample.timestamp, sample)
    })
    const merged = [...byTimestamp.values()].sort((left, right) => left.timestamp - right.timestamp)
    if (merged.length) state.gpuHistory.set(Number(index), merged)
  })
  state.gpuHistory.forEach((samples, index) => {
    const recent = samples.filter((sample) => sample.timestamp >= cutoff)
    if (recent.length) state.gpuHistory.set(index, recent)
    else state.gpuHistory.delete(index)
  })
  persistGpuSession()
}

function gpuMetricMaximum(metric, gpu, history) {
  const observed = Math.max(1, ...history.map((sample) => sample[metric.key] || 0))
  if (['utilization', 'memory_utilization', 'temperature', 'fan_speed'].includes(metric.key)) return 100
  if (metric.key === 'memory_used' && Number.isFinite(gpu.memory_total)) return Math.max(1, gpu.memory_total)
  if (metric.key === 'power' && Number.isFinite(gpu.power_limit)) return Math.max(1, gpu.power_limit)
  if (metric.key === 'graphics_clock' && Number.isFinite(gpu.graphics_clock_max)) return Math.max(1, gpu.graphics_clock_max)
  if (metric.key === 'memory_clock' && Number.isFinite(gpu.memory_clock_max)) return Math.max(1, gpu.memory_clock_max)
  return Math.ceil(observed * 1.1)
}

function formatGpuMetric(metric, value) {
  if (!Number.isFinite(value)) return 'N/A'
  if (['utilization', 'memory_utilization', 'fan_speed'].includes(metric.key)) return `${Math.round(value)}%`
  if (metric.key === 'memory_used') return `${(value / 1024).toFixed(1)} GB`
  if (metric.key === 'temperature') return `${Math.round(value)} C`
  if (metric.key === 'power') return `${Math.round(value)} W`
  return `${Math.round(value)} MHz`
}

function attachGpuChartHover(plot, samples, metric, now) {
  const line = element('div', 'gpu-hover-line')
  const tooltip = element('div', 'gpu-hover-tooltip')
  line.hidden = true
  tooltip.hidden = true
  plot.append(line, tooltip)
  plot.addEventListener('pointermove', (event) => {
    state.gpuHovering = true
    const bounds = plot.getBoundingClientRect()
    const offset = Math.min(bounds.width, Math.max(0, event.clientX - bounds.left))
    const ratio = bounds.width ? offset / bounds.width : 0
    const targetTime = now - state.gpuWindowMs + (ratio * state.gpuWindowMs)
    const nearest = samples.reduce((best, sample) => {
      if (!best) return sample
      return Math.abs(sample.timestamp - targetTime) < Math.abs(best.timestamp - targetTime) ? sample : best
    }, null)
    const tolerance = Math.max(1500, state.gpuWindowMs * 10 / Math.max(1, bounds.width))
    const hasSample = nearest && Math.abs(nearest.timestamp - targetTime) <= tolerance
    const shownTime = new Date(hasSample ? nearest.timestamp : targetTime).toLocaleTimeString()
    tooltip.textContent = hasSample ? `${formatGpuMetric(metric, nearest[metric.key])} / ${shownTime}` : `No sample / ${shownTime}`
    const percent = ratio * 100
    line.style.left = `${percent}%`
    tooltip.style.left = `${percent}%`
    tooltip.classList.toggle('align-start', percent < 18)
    tooltip.classList.toggle('align-end', percent > 82)
    line.hidden = false
    tooltip.hidden = false
  })
  plot.addEventListener('pointerleave', () => {
    state.gpuHovering = false
    line.hidden = true
    tooltip.hidden = true
    renderGpuMonitor(state.gpuStats)
  })
}

function createGpuMetricChart(metric, gpu, history, now) {
  const current = gpu[metric.key]
  if (!Number.isFinite(current)) return null
  const samples = history.filter((sample) => Number.isFinite(sample[metric.key]))
  const maximum = gpuMetricMaximum(metric, gpu, samples)
  const average = samples.length ? samples.reduce((total, sample) => total + sample[metric.key], 0) / samples.length : current
  const peak = samples.length ? Math.max(...samples.map((sample) => sample[metric.key])) : current
  const chart = element('div', 'gpu-metric-chart')
  chart.style.setProperty('--chart-color', metric.color)
  const chartScale = element('div', 'gpu-chart-scale')
  chartScale.append(element('span', '', metric.label), element('strong', '', formatGpuMetric(metric, current)))
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('class', 'gpu-sparkline')
  svg.setAttribute('viewBox', '0 0 300 70')
  svg.setAttribute('preserveAspectRatio', 'none')
  svg.setAttribute('role', 'img')
  svg.setAttribute('aria-label', `${metric.label} history, average ${formatGpuMetric(metric, average)}, peak ${formatGpuMetric(metric, peak)}`)
  addGpuChartGrid(svg, 300, 70)
  const points = gpuHistoryPoints(samples, now, state.gpuWindowMs, metric.key, maximum)
  if (samples.length > 1) {
    const pointList = points.split(' ')
    const area = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
    area.setAttribute('class', 'gpu-chart-area')
    area.setAttribute('points', `${pointList[0].split(',')[0]},70 ${points} ${pointList.at(-1).split(',')[0]},70`)
    svg.append(area)
  }
  const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline')
  polyline.setAttribute('class', 'gpu-chart-line')
  polyline.setAttribute('points', points)
  svg.append(polyline)
  if (samples.length) {
    const latest = points.split(' ').at(-1).split(',')
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
    marker.setAttribute('class', 'gpu-chart-marker')
    marker.setAttribute('cx', latest[0])
    marker.setAttribute('cy', latest[1])
    marker.setAttribute('r', '2.5')
    svg.append(marker)
  }
  const plot = element('div', 'gpu-chart-plot')
  plot.append(svg)
  attachGpuChartHover(plot, samples, metric, now)
  const axis = element('div', 'gpu-chart-axis')
  axis.append(
    element('span', '', state.gpuWindowMs === 60 * 1000 ? '1 minute' : '10 minutes'),
    element('span', '', `Average ${formatGpuMetric(metric, average)} · Peak ${formatGpuMetric(metric, peak)}`),
  )
  chart.append(chartScale, plot, axis)
  return chart
}

function renderGpuMonitor(gpus) {
  const output = $('#gpu-output')
  const monitor = element('div', 'gpu-monitor')
  const heading = element('div', 'gpu-monitor-heading')
  const title = element('div', 'gpu-monitor-title')
  const icon = element('i', 'icon-activity')
  title.append(icon, element('strong', '', 'GPU telemetry'))
  const windowControl = element('div', 'gpu-window-control')
  windowControl.setAttribute('role', 'group')
  windowControl.setAttribute('aria-label', 'GPU history window')
  ;[[60 * 1000, '1 min'], [10 * 60 * 1000, '10 min']].forEach(([windowMs, label]) => {
    const button = element('button', windowMs === state.gpuWindowMs ? 'active' : '', label)
    button.type = 'button'
    button.setAttribute('aria-pressed', String(windowMs === state.gpuWindowMs))
    button.addEventListener('click', () => {
      state.gpuWindowMs = windowMs
      persistUiState()
      renderGpuMonitor(state.gpuStats)
    })
    windowControl.append(button)
  })
  heading.append(title, windowControl)
  monitor.append(heading)
  if (!gpus.length) {
    monitor.append(element('div', 'gpu-monitor-muted', 'GPU telemetry unavailable.'))
    output.replaceChildren(monitor)
    return
  }

  const grid = element('div', 'gpu-card-grid')
  gpus.forEach((gpu) => {
    const now = Date.now()
    const history = (state.gpuHistory.get(gpu.index) || []).filter((sample) => sample.timestamp >= now - state.gpuWindowMs)
    const card = element('article', 'gpu-card')
    const cardHead = element('div', 'gpu-card-head')
    cardHead.append(element('strong', '', `GPU ${gpu.index}`), element('span', '', gpu.name))
    const metrics = element('div', 'gpu-metrics-grid')
    GPU_METRICS.forEach((metric) => {
      const chart = createGpuMetricChart(metric, gpu, history, now)
      if (chart) metrics.append(chart)
    })
    const details = element('div', 'gpu-live-details')
    if (gpu.performance_state) details.append(element('span', '', `State ${gpu.performance_state}`))
    if (Number.isFinite(gpu.pcie_generation) && Number.isFinite(gpu.pcie_width)) details.append(element('span', '', `PCIe Gen ${gpu.pcie_generation} x${gpu.pcie_width}`))
    if (Number.isFinite(gpu.power_limit)) details.append(element('span', '', `Power limit ${Math.round(gpu.power_limit)} W`))
    card.append(cardHead, metrics, details)
    grid.append(card)
  })
  monitor.append(grid)
  output.replaceChildren(monitor)
}

async function refreshGpuMonitor() {
  if (state.gpuRefreshActive) return
  state.gpuRefreshActive = true
  try {
    const payload = await fetchJson('/system/gpu')
    state.gpuStats = Array.isArray(payload.gpus) ? payload.gpus : []
    mergeGpuHistory(payload.history)
    if (!state.gpuHovering) renderGpuMonitor(state.gpuStats)
  } catch {
    if (!state.gpuHovering) renderGpuMonitor(state.gpuStats)
  } finally {
    state.gpuRefreshActive = false
  }
}

function startGpuMonitor() {
  if (state.gpuTimer || document.hidden) return
  if (state.gpuStats.length) renderGpuMonitor(state.gpuStats)
  refreshGpuMonitor()
  state.gpuTimer = setInterval(refreshGpuMonitor, GPU_POLL_INTERVAL_MS)
}

function stopGpuMonitor() {
  clearInterval(state.gpuTimer)
  state.gpuTimer = null
}

async function refreshSystemStatus() {
  try {
    const status = await fetchJson('/tts/status')
    $('#readiness-output').textContent = JSON.stringify(status, null, 2)
  } catch (error) {
    $('#readiness-output').textContent = errorMessage(error)
  }
}

async function refreshSystem() {
  await Promise.all([refreshSystemStatus(), refreshGpuMonitor()])
}

function activateTab(tab) {
  state.activeTab = tab
  const workflowActive = ['generate', 'clone', 'stream'].includes(tab)
  $('.workspace').dataset.view = tab
  $$('.tab-button').forEach((button) => {
    const active = button.dataset.tab === tab
    button.classList.toggle('active', active)
    button.setAttribute('aria-selected', String(active))
  })
  $$('.tab-panel').forEach((panel) => { panel.hidden = panel.dataset.panel !== tab })
  $('#inference-settings').hidden = !workflowActive
  $('#composer').hidden = !workflowActive
  updateWorkflowControls()
  if (tab !== 'clone') referenceRecorder.stop()
  else requestAnimationFrame(() => referenceRecorder.refresh())
  stopGpuMonitor()
  if (tab === 'api') refreshApi().catch((error) => showToast(errorMessage(error)))
  if (tab === 'system') {
    refreshSystemStatus()
    startGpuMonitor()
  }
  refreshFormatOptions()
  persistUiState()
}

function setHeaderCollapsed(collapsed) {
  state.headerCollapsed = collapsed
  document.documentElement.dataset.headerCollapsed = String(collapsed)
  $('#brand-hero').dataset.collapsed = String(collapsed)
  const button = $('#hero-toggle')
  button.setAttribute('aria-expanded', String(!collapsed))
  button.setAttribute('aria-label', collapsed ? 'Expand header' : 'Collapse header')
  button.title = collapsed ? 'Expand header' : 'Collapse header'
  button.innerHTML = `<i class="icon-chevron-${collapsed ? 'down' : 'up'}"></i>`
  persistUiState()
}

function bindRangeInputs() {
  $$('[data-value-input]').forEach((range) => {
    const number = $(`#${range.dataset.valueInput}`)
    range.addEventListener('input', () => { number.value = range.value })
  })
  $$('[data-range-input]').forEach((number) => {
    const range = $(`#${number.dataset.rangeInput}`)
    number.addEventListener('input', () => { range.value = number.value })
  })
}

function resetControls() {
  $('#guidance').value = 2
  $('#guidance-slider').value = 2
  $('#steps').value = state.defaults.inference_timesteps || 10
  $('#steps-slider').value = state.defaults.inference_timesteps || 10
  $('#normalize-text').checked = false
  $('#normalize-loudness').checked = state.defaults.normalize_loudness ?? true
  $('#denoise').checked = false
  $('#seed').value = state.defaults.seed ?? 42
  $('#randomize-seed').checked = state.defaults.randomize_seed ?? true
  updateSeedState()
}

async function initialize() {
  const [defaults, status, languages, formats, streamFormats, profiles] = await Promise.all([
    fetchJson('/tts/defaults'),
    fetchJson('/tts/status'),
    fetchJson('/tts/languages'),
    fetchJson('/tts/formats'),
    fetchJson('/tts/stream-formats'),
    fetchJson('/tts/voice-profiles'),
  ])
  state.defaults = defaults
  state.status = status
  state.formats = formats.formats || {}
  state.streamFormats = streamFormats.formats || {}
  state.profiles = profiles.data || []

  populateSelect($('#language'), languages.languages.map((language) => ({ value: language, label: language })), defaults.language)
  populateSelect($('#device'), status.hardware || [{ value: 'auto', label: 'Auto' }, { value: 'cpu', label: 'CPU' }], defaults.device)
  $('#profile-count').textContent = String(state.profiles.length)
  renderVoiceProfileSelect()
  renderProfileList()
  resetControls()
  $('#denoise').disabled = !status.load_denoiser
  $('#transcribe-reference').disabled = !status.load_asr
  const timestampsAvailable = Boolean(status.timestamps?.available)
  $('#generate-timestamps').disabled = !timestampsAvailable
  $('#timestamp-state').textContent = timestampsAvailable ? 'Ready' : 'Unavailable'
  $('#timestamp-state').dataset.state = timestampsAvailable ? 'available' : 'unavailable'
  updateTimestampState()
  $('#runtime-badge').dataset.state = 'ready'
  $('#runtime-state').textContent = `${status.backend === 'nano' ? 'Nano' : 'Native'} backend ready`
  $('#runtime-model').textContent = `${status.model_id} · ${status.runtime}`
  setStatus('Ready', 'success')
  updateVoiceDesignState()
  updateCloneConditioning()
  refreshFormatOptions()

  setHeaderCollapsed(state.headerCollapsed)
  activateTab(state.activeTab)
}

$('#text-input').addEventListener('input', updateMetrics)
$$('.input-type-control button').forEach((button) => button.addEventListener('click', () => setInputType(button.dataset.inputType)))
$('#control-input').addEventListener('input', updateVoiceDesignState)
$('#reference-text').addEventListener('input', updateCloneConditioning)
$('#voice-profile').addEventListener('change', updateVoiceProfileState)
$('#randomize-seed').addEventListener('change', updateSeedState)
$('#generate-timestamps').addEventListener('change', updateTimestampState)
$('#sample-button').addEventListener('click', () => {
  if (state.inputType === 'text') state.sampleIndex = (state.sampleIndex + 1) % SAMPLE_TEXTS.length
  $('#text-input').value = inputSample(state.inputType)
  state.inputDrafts[state.inputType] = $('#text-input').value
  updateMetrics()
})
$('#reference-audio').addEventListener('change', (event) => {
  const file = event.target.files[0]
  if (file) referenceAudio.load(file, file.name)
})
for (const eventName of ['dragenter', 'dragover']) {
  $('#reference-audio-drop').addEventListener(eventName, (event) => {
    event.preventDefault()
    $('#reference-audio-drop').classList.add('dragging')
  })
}
for (const eventName of ['dragleave', 'drop']) {
  $('#reference-audio-drop').addEventListener(eventName, (event) => {
    event.preventDefault()
    $('#reference-audio-drop').classList.remove('dragging')
  })
}
$('#reference-audio-drop').addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0]
  if (file) referenceAudio.load(file, file.name)
})
$('#profile-name').addEventListener('input', updateProfileNamePreview)
$('#profile-filter').addEventListener('input', renderProfileList)
$('#profile-audio').addEventListener('change', (event) => loadProfileAudio(event.target.files[0]))
for (const eventName of ['dragenter', 'dragover']) {
  $('#profile-audio-drop').addEventListener(eventName, (event) => {
    event.preventDefault()
    event.currentTarget.classList.add('dragging')
  })
}
for (const eventName of ['dragleave', 'drop']) {
  $('#profile-audio-drop').addEventListener(eventName, (event) => {
    event.preventDefault()
    event.currentTarget.classList.remove('dragging')
  })
}
$('#profile-audio-drop').addEventListener('drop', (event) => loadProfileAudio(event.dataTransfer.files[0]))
$('#voice-form').addEventListener('submit', async (event) => {
  event.preventDefault()
  const formElement = event.currentTarget
  const button = $('button[type="submit"]', formElement)
  const name = normalizedProfileName($('#profile-name').value)
  const file = profileAudio.currentFile()
  if (!name) return showToast('Enter a voice name.')
  if (!file && !state.editingProfile) return showToast('Choose or drop a reference audio sample.')
  button.disabled = true
  try {
    const saved = await saveProfileRequest({
      name,
      profileType: 'cloned',
      description: $('#profile-description').value.trim(),
      file,
      refText: $('#profile-transcript').value.trim(),
      editing: Boolean(state.editingProfile),
    })
    resetProfileEditor()
    await refreshProfiles(saved.id)
    showToast(`Saved ${saved.id}.`, 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})
$('#profile-edit-cancel').addEventListener('click', resetProfileEditor)
$('#save-designed-voice').addEventListener('click', () => openQuickSaveDialog('designed'))
$('#save-cloned-voice').addEventListener('click', () => openQuickSaveDialog('cloned'))
$('#quick-save-form').addEventListener('submit', async (event) => {
  event.preventDefault()
  const profileType = state.quickSaveType
  const name = normalizedProfileName($('#quick-profile-name').value)
  if (!profileType || !name) return showToast('Enter a voice name.')
  const button = $('button[type="submit"]', event.currentTarget)
  button.disabled = true
  try {
    const refText = profileType === 'cloned' ? $('#reference-text').value.trim() : ''
    const saved = await saveProfileRequest({
      name,
      profileType,
      description: $('#quick-profile-description').value.trim(),
      file: profileType === 'cloned' ? referenceAudio.currentFile() : null,
      refText,
      control: profileType === 'designed'
        ? $('#control-input').value.trim()
        : (refText ? '' : $('#clone-control-input').value.trim()),
    })
    closeQuickSaveDialog()
    await refreshProfiles(saved.id)
    showToast(`Saved ${saved.id}.`, 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})
$('#save-profile-close').addEventListener('click', closeQuickSaveDialog)
$('#save-profile-cancel').addEventListener('click', closeQuickSaveDialog)
$('#save-profile-dialog').addEventListener('click', (event) => { if (event.target === event.currentTarget) closeQuickSaveDialog() })
$('#delete-profile-close').addEventListener('click', closeDeleteProfileDialog)
$('#delete-profile-cancel').addEventListener('click', closeDeleteProfileDialog)
$('#delete-profile-dialog').addEventListener('click', (event) => { if (event.target === event.currentTarget) closeDeleteProfileDialog() })
$('#delete-profile-confirm').addEventListener('click', async (event) => {
  const profile = state.pendingDeleteProfile
  if (!profile) return
  const button = event.currentTarget
  button.disabled = true
  try {
    await fetchJson(`/tts/voice-profiles/${encodeURIComponent(profile.id)}`, { method: 'DELETE' })
    closeDeleteProfileDialog()
    await refreshProfiles()
    showToast(`Deleted ${profile.id}.`, 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})
$('#generate-button').addEventListener('click', () => generateAudio('generate'))
$('#clone-button').addEventListener('click', () => generateAudio('clone'))
$('#stream-start').addEventListener('click', streamAudio)
$('#stream-stop').addEventListener('click', () => {
  state.streamAbort?.abort()
  state.streamPlayback?.stop()
})
$('#transcribe-reference').addEventListener('click', transcribeReference)
$('#reset-controls').addEventListener('click', resetControls)
$('#api-refresh').addEventListener('click', () => refreshApi().catch((error) => showToast(errorMessage(error))))
$('#system-refresh').addEventListener('click', refreshSystem)
$('#purge-models').addEventListener('click', async () => {
  try {
    const result = await fetchJson('/tts/purge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
    showToast(`Purged ${result.purged.length} model cache entries.`, 'success')
    refreshSystem()
  } catch (error) {
    showToast(errorMessage(error))
  }
})
$('#hero-toggle').addEventListener('click', () => setHeaderCollapsed(!state.headerCollapsed))
$$('.tab-button').forEach((button) => button.addEventListener('click', () => activateTab(button.dataset.tab)))

bindRangeInputs()
restoreSessionState()
$('#text-input').dataset.inputType = 'text'
state.inputDrafts.text = $('#text-input').value
setInputType(state.inputType)
initialize().catch((error) => {
  $('#runtime-badge').dataset.state = 'error'
  $('#runtime-state').textContent = 'Service unavailable'
  $('#runtime-model').textContent = errorMessage(error)
  setStatus(errorMessage(error), 'error')
})

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopGpuMonitor()
  else if (state.activeTab === 'system') startGpuMonitor()
})
window.addEventListener('beforeunload', () => {
  stopGpuMonitor()
  state.streamAbort?.abort()
  state.streamPlayback?.stop()
  clearInterval(state.activityTimer)
  streamWaveform.hide()
  referenceRecorder.stop()
  generateOutput.destroy()
  cloneOutput.destroy()
  streamOutput.destroy()
  referenceAudio.destroy()
  profileAudio.destroy()
})
