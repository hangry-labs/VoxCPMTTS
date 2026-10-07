import { AudioEditor } from './audio-editor.js'
import { AudioRecorder } from './audio-recorder.js'

const $ = (selector, root = document) => root.querySelector(selector)
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)]

const UI_SESSION_KEY = 'voxcpmtts-ui-state-v1'
const SAMPLE_TEXTS = [
  'VoxCPM2 generates natural multilingual speech with voice design and cloning.',
  'A steady voice can make technical information easier to understand.',
  'Good morning. The latest build is ready for a careful listening test.',
  'This reference voice can speak new text while preserving its character and pacing.',
]
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

const state = {
  activeTab: 'generate',
  headerCollapsed: document.documentElement.dataset.headerCollapsed === 'true',
  streamAbort: null,
  sampleIndex: 0,
  gpuTimer: null,
  formats: {},
  streamFormats: {},
  defaults: {},
  status: {},
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
const referenceAudio = new AudioEditor($('#reference-audio-preview'), {
  label: 'Reference preview',
  emptyTitle: 'No reference selected',
  emptyDescription: 'Choose or record a sample',
  labels: AUDIO_EDITOR_LABELS,
  onChange: (file) => {
    $('#reference-audio-drop').classList.toggle('has-file', Boolean(file))
    $('#reference-audio-name').textContent = file?.name || 'WAV, MP3, FLAC, OGG, or M4A'
  },
})
const referenceRecorder = new AudioRecorder({
  button: $('#reference-record-toggle'),
  status: $('#reference-record-state'),
  labels: RECORDER_LABELS,
  onFile: (file) => referenceAudio.load(file, file.name),
  onError: (error) => showToast(errorMessage(error)),
})

for (const editor of [generateOutput, streamOutput, referenceAudio]) {
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
    }))
  } catch {
    // Browser storage may be unavailable in privacy-restricted sessions.
  }
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

function populateSelect(select, options, selected) {
  select.replaceChildren(...options.map(({ value, label }) => {
    const option = document.createElement('option')
    option.value = value
    option.textContent = label
    return option
  }))
  if (options.some((option) => option.value === selected)) select.value = selected
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

function updateVoiceMode() {
  const mode = $('#voice-mode').value
  const usesControl = mode === 'design' || mode === 'clone'
  const usesReference = mode === 'clone' || mode === 'guided'
  $('#design-panel').hidden = !usesControl
  $('#clone-panel').hidden = !usesReference
  $('#guided-fields').hidden = mode !== 'guided'
  $('#control-input').disabled = !usesControl
  $('#reference-audio').disabled = !usesReference
  $('#control-input').previousElementSibling.textContent = mode === 'clone' ? 'Clone direction' : 'Voice description'
  $('#denoise').disabled = !usesReference || !state.status.load_denoiser
}

function buildPayload({ streaming = false } = {}) {
  const mode = $('#voice-mode').value
  const outputFormat = $('#output-format').value
  const payload = {
    text: $('#text-input').value.trim(),
    language: $('#language').value || 'English',
    voice: mode === 'clone' || mode === 'guided' ? 'reference' : 'auto',
    control: mode === 'design' || mode === 'clone' ? ($('#control-input').value.trim() || null) : null,
    ref_text: mode === 'guided' ? ($('#reference-text').value.trim() || null) : null,
    cfg_value: Number($('#guidance').value),
    inference_timesteps: Number($('#steps').value),
    normalize: $('#normalize').checked,
    denoise: $('#denoise').checked,
    device: $('#device').value,
    output_format: outputFormat,
  }
  if (streaming) payload.stream_format = outputFormat
  return payload
}

async function requestAudio({ streaming = false, signal } = {}) {
  const payload = buildPayload({ streaming })
  if (!payload.text) throw new Error('Enter text to synthesize.')
  const mode = $('#voice-mode').value
  const needsReference = mode === 'clone' || mode === 'guided'
  const reference = referenceAudio.currentFile()
  if (needsReference && !reference) throw new Error('Choose or record reference audio.')
  if (mode === 'guided' && !payload.ref_text) throw new Error('Enter the reference transcript.')

  const route = streaming ? '/tts/stream' : '/tts/generate'
  let response
  if (needsReference) {
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
  return {
    blob: await response.blob(),
    extension: response.headers.get('X-VoxCPM-Format') || payload.output_format,
  }
}

function setGenerationBusy(active, streaming = false) {
  const button = streaming ? $('#stream-start') : $('#generate-button')
  button.disabled = active
  if (streaming) {
    $('#stream-stop').disabled = !active
    $('#stream-progress').hidden = !active
  }
}

async function generateAudio() {
  setGenerationBusy(true)
  setStatus('Generating audio')
  try {
    const { blob, extension } = await requestAudio()
    await generateOutput.load(blob, `voxcpmtts.${extension}`)
    setStatus('Generation complete', 'success')
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    showToast(errorMessage(error))
  } finally {
    setGenerationBusy(false)
  }
}

async function streamAudio() {
  const controller = new AbortController()
  state.streamAbort = controller
  setGenerationBusy(true, true)
  setStatus('Streaming audio')
  try {
    const { blob, extension } = await requestAudio({ streaming: true, signal: controller.signal })
    await streamOutput.load(blob, `voxcpmtts-stream.${extension}`)
    setStatus('Stream complete', 'success')
  } catch (error) {
    if (error.name === 'AbortError') setStatus('Stream stopped')
    else {
      setStatus(errorMessage(error), 'error')
      showToast(errorMessage(error))
    }
  } finally {
    state.streamAbort = null
    setGenerationBusy(false, true)
  }
}

async function transcribeReference() {
  const file = referenceAudio.currentFile()
  if (!file) return showToast('Choose or record reference audio first.')
  const button = $('#transcribe-reference')
  button.disabled = true
  setStatus('Transcribing reference')
  try {
    const form = new FormData()
    form.append('reference_audio', file, file.name)
    form.append('language', 'auto')
    const result = await fetchJson('/tts/transcribe-upload', { method: 'POST', body: form })
    $('#reference-text').value = result.text || ''
    setStatus('Reference transcript ready', 'success')
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    showToast(errorMessage(error))
  } finally {
    button.disabled = !state.status.load_asr
  }
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

function gpuMetric(label, value) {
  return `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`
}

function renderGpu(payload) {
  const container = $('#gpu-output')
  if (!payload.gpus?.length) {
    container.innerHTML = '<div class="gpu-monitor-muted">No NVIDIA GPU telemetry available.</div>'
    return
  }
  container.innerHTML = `<div class="gpu-monitor"><div class="gpu-monitor-heading"><div class="gpu-monitor-title"><i class="icon-activity"></i><strong>GPU telemetry</strong></div></div><div class="gpu-card-grid">${payload.gpus.map((gpu) => {
    const memory = gpu.memory_used == null ? '--' : `${gpu.memory_used} / ${gpu.memory_total} MiB`
    return `<article class="gpu-card"><div class="gpu-card-head"><strong>GPU ${escapeHtml(gpu.index)}</strong><span>${escapeHtml(gpu.name)}</span></div><div class="gpu-live-details">${gpuMetric('Utilization', gpu.utilization == null ? '--' : `${gpu.utilization}%`)}${gpuMetric('VRAM', memory)}${gpuMetric('Temperature', gpu.temperature == null ? '--' : `${gpu.temperature}°C`)}${gpuMetric('Power', gpu.power == null ? '--' : `${gpu.power} W`)}</div></article>`
  }).join('')}</div></div>`
}

async function refreshSystem() {
  try {
    const [status, gpu] = await Promise.all([fetchJson('/tts/status'), fetchJson('/system/gpu')])
    $('#readiness-output').textContent = JSON.stringify(status, null, 2)
    renderGpu(gpu)
  } catch (error) {
    $('#readiness-output').textContent = errorMessage(error)
  }
}

function activateTab(tab) {
  state.activeTab = tab
  const workflowActive = tab === 'generate' || tab === 'stream'
  $('.workspace').dataset.view = tab
  $$('.tab-button').forEach((button) => {
    const active = button.dataset.tab === tab
    button.classList.toggle('active', active)
    button.setAttribute('aria-selected', String(active))
  })
  $$('.tab-panel').forEach((panel) => { panel.hidden = panel.dataset.panel !== tab })
  $('#inference-settings').hidden = !workflowActive
  $('#composer').hidden = !workflowActive
  if (workflowActive) updateVoiceMode()
  else {
    $('#design-panel').hidden = true
    $('#clone-panel').hidden = true
  }
  clearInterval(state.gpuTimer)
  state.gpuTimer = null
  if (tab === 'api') refreshApi().catch((error) => showToast(errorMessage(error)))
  if (tab === 'system') {
    refreshSystem()
    state.gpuTimer = setInterval(refreshSystem, 2000)
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
  $('#normalize').checked = false
  $('#denoise').checked = false
}

async function initialize() {
  const [defaults, status, languages, formats, streamFormats] = await Promise.all([
    fetchJson('/tts/defaults'),
    fetchJson('/tts/status'),
    fetchJson('/tts/languages'),
    fetchJson('/tts/formats'),
    fetchJson('/tts/stream-formats'),
  ])
  state.defaults = defaults
  state.status = status
  state.formats = formats.formats || {}
  state.streamFormats = streamFormats.formats || {}

  populateSelect($('#language'), languages.languages.map((language) => ({ value: language, label: language })), defaults.language)
  populateSelect($('#device'), status.hardware || [{ value: 'auto', label: 'Auto' }, { value: 'cpu', label: 'CPU' }], defaults.device)
  resetControls()
  $('#denoise').disabled = !status.load_denoiser
  $('#transcribe-reference').disabled = !status.load_asr
  $('#runtime-badge').dataset.state = 'ready'
  $('#runtime-state').textContent = `${status.backend === 'nano' ? 'Nano' : 'Native'} backend ready`
  $('#runtime-model').textContent = `${status.model_id} · ${status.runtime}`
  setStatus('Ready', 'success')
  updateVoiceMode()
  refreshFormatOptions()

  let saved = null
  try { saved = JSON.parse(sessionStorage.getItem(UI_SESSION_KEY) || 'null') } catch {}
  setHeaderCollapsed(Boolean(saved?.headerCollapsed))
  activateTab(['generate', 'stream', 'api', 'system'].includes(saved?.activeTab) ? saved.activeTab : 'generate')
}

$('#voice-mode').addEventListener('change', updateVoiceMode)
$('#text-input').addEventListener('input', updateMetrics)
$('#sample-button').addEventListener('click', () => {
  state.sampleIndex = (state.sampleIndex + 1) % SAMPLE_TEXTS.length
  $('#text-input').value = SAMPLE_TEXTS[state.sampleIndex]
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
$('#generate-button').addEventListener('click', generateAudio)
$('#stream-start').addEventListener('click', streamAudio)
$('#stream-stop').addEventListener('click', () => state.streamAbort?.abort())
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
updateMetrics()
initialize().catch((error) => {
  $('#runtime-badge').dataset.state = 'error'
  $('#runtime-state').textContent = 'Service unavailable'
  $('#runtime-model').textContent = errorMessage(error)
  setStatus(errorMessage(error), 'error')
})

window.addEventListener('beforeunload', () => {
  clearInterval(state.gpuTimer)
  state.streamAbort?.abort()
  referenceRecorder.stop()
  generateOutput.destroy()
  streamOutput.destroy()
  referenceAudio.destroy()
})
