import { browserLanguage, initializeI18n, languageLabel, t } from './i18n.js'
import { AudioEditor } from './audio-editor.js?v=waveform-hitbox'
import { AudioFinisher } from './audio-finisher.js?v=long-form-audio'
import { AudioRecorder } from './audio-recorder.js?v=voice-library'
import { DialogueScriptLibrary } from './dialogue-script-library.js?v=app-organization'
import { GenerationToolbar } from './generation-toolbar.js?v=workflow-controls'
import { GpuMonitor } from './gpu-monitor.js?v=app-organization'
import { MagicEditor } from './magic-editor.js?v=removable-chips'
import { MagicTakeStudio } from './magic-takes.js?v=turn-studio'
import { IncrementalAudioPlayback, StreamWaveform } from './streaming-player.js?v=app-organization'
import { VersionCheck } from './version-check.js?v=published-builds'

await initializeI18n()

const $ = (selector, root = document) => root.querySelector(selector)
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)]

const UI_SESSION_KEY = 'voxcpmtts-ui-state-v1'
const MAX_GENERATED_REFERENCE_CHARACTERS = 320
const POST_PROCESSING_PRESETS = {
  clean: { pitch_semitones: 0, speed_factor: 1, noise_reduction_db: 1, bass_db: 0, presence_db: 0.5, dynamics: 15, normalize_loudness: true },
  studio: { pitch_semitones: 0, speed_factor: 1, noise_reduction_db: 2, bass_db: 1, presence_db: 1, dynamics: 35, normalize_loudness: true },
}
const SAMPLE_TEXTS = [
  t('samples.first', {}, 'VoxCPM2 generates natural multilingual speech with voice design and cloning.'),
  t('samples.second', {}, 'A steady voice can make technical information easier to understand.'),
  t('samples.third', {}, 'Good morning. The latest build is ready for a careful listening test.'),
  t('samples.fourth', {}, 'This reference voice can speak new text while preserving its character and pacing.'),
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
  <voice name="Host" h:direction="Energetic">Welcome to the show. What are we exploring today?</voice>
  <voice name="Guest" h:direction="Calm and authoritative">We are testing a complete multi-speaker discussion from one document.</voice>
</speak>`,
  ],
}
const AUDIO_EDITOR_LABELS = {
  noAudio: t('audio.noAudio', {}, 'No audio selected'),
  download: t('audio.download', {}, 'Download audio'),
  share: t('audio.share', {}, 'Share audio'),
  remove: t('audio.remove', {}, 'Remove audio'),
  mute: t('audio.mute', {}, 'Mute or unmute'),
  volume: t('audio.volume', {}, 'Volume'),
  playbackSpeed: t('audio.playbackSpeed', {}, 'Playback speed'),
  backward: t('audio.backward', {}, 'Seek backward 5 seconds'),
  play: t('audio.play', {}, 'Play'),
  pause: t('audio.pause', {}, 'Pause'),
  forward: t('audio.forward', {}, 'Seek forward 5 seconds'),
  restart: t('audio.restart', {}, 'Return to start'),
  trim: t('audio.trim', {}, 'Select and trim audio'),
  cancel: t('common.cancel', {}, 'Cancel'),
  applySelection: t('audio.applySelection', {}, 'Apply selection'),
}
const RECORDER_LABELS = {
  record: t('record.record', {}, 'Record'),
  stop: t('record.stop', {}, 'Stop recording'),
  ready: t('record.ready', {}, 'Ready to record'),
  recording: t('record.recording', {}, 'Recording {time}'),
  processing: t('record.processing', {}, 'Preparing recording'),
  readyWithTime: t('record.readyWithTime', {}, 'Recording ready · {time}'),
  unavailable: t('record.unavailable', {}, 'Microphone recording requires a supported browser and secure connection.'),
}
const state = {
  activeTab: 'generate',
  generationMode: 'generate',
  inputType: 'magic',
  inputDrafts: { text: null, ssml: null, 'ssml-h': null },
  headerCollapsed: document.documentElement.dataset.headerCollapsed === 'true',
  streamAbort: null,
  streamPlayback: null,
  sampleIndex: 0,
  formats: {},
  streamFormats: {},
  ssmlCapabilities: {},
  defaults: {},
  status: {},
  profiles: [],
  designSource: 'reference',
  cloneMode: 'reference',
  loadingProfile: false,
  lastCloneGeneration: null,
  selectedCloneVersion: 'original',
  quickSaveType: null,
  editingProfile: null,
  portraitFile: null,
  portraitObjectUrl: null,
  portraitRemoved: false,
  portraitTargetProfile: null,
  magicCharacterBlockId: null,
  magicCharacterPortraitFile: null,
  magicCharacterPortraitUrl: null,
  pendingDeleteProfile: null,
  activityTimer: null,
  magicAssemblyTimer: null,
  magicAssemblyRevision: 0,
  magicAssemblyAbort: null,
  magicGenerating: false,
}

const generationToolbar = new GenerationToolbar($('#generation-toolbar'), {
  randomLabel: t('generation.randomSeed', {}, 'Randomize'),
})
const magicTakeStudio = new MagicTakeStudio({ responseError })
const magicEditor = new MagicEditor($('#magic-editor-shell'), {
  onChange: handleMagicChange,
  onPreview: generateMagicPreview,
  onSaveCharacter: openMagicCharacterDialog,
  onError: (error) => showToast(errorMessage(error)),
})
state.inputDrafts['ssml-h'] = magicEditor.toSSMLH()

const generateOutput = new AudioEditor($('#generate-output'), {
  label: t('output.generated', {}, 'Generated audio'),
  emptyTitle: t('output.emptyTitle', {}, 'Audio output'),
  emptyDescription: t('output.ready', {}, 'Ready for synthesis'),
  labels: AUDIO_EDITOR_LABELS,
})
const generateFinisher = new AudioFinisher($('#generate-finishing'), {
  audioEditorLabels: AUDIO_EDITOR_LABELS,
  labels: {
    processed: t('finishing.processedAudio', {}, 'Processed audio'),
    emptyTitle: t('output.emptyTitle', {}, 'Audio output'),
    ready: t('finishing.ready', {}, 'Ready for processing'),
    ffmpeg: t('finishing.ffmpegStudio', {}, 'FFmpeg Studio'),
    signalsmith: t('finishing.signalsmithVoice', {}, 'Signalsmith Voice'),
    generateFirst: t('errors.generateAudioBeforeProcess', {}, 'Generate audio before processing it.'),
    processing: t('status.processingAudio', {}, 'Finishing audio with {method}'),
    complete: t('status.processedAudioReady', {}, 'Processed audio ready'),
  },
  responseError,
  onError: (error) => showToast(errorMessage(error)),
})
const streamOutput = new AudioEditor($('#stream-output'), {
  label: t('output.streamed', {}, 'Streamed audio'),
  emptyTitle: t('output.emptyTitle', {}, 'Audio output'),
  emptyDescription: t('output.readyStream', {}, 'Ready for streaming'),
  labels: AUDIO_EDITOR_LABELS,
})
const cloneOutput = new AudioEditor($('#clone-output'), {
  label: t('output.designed', {}, 'Designed voice'),
  emptyTitle: t('output.emptyTitle', {}, 'Audio output'),
  emptyDescription: t('output.readyDesign', {}, 'Ready for voice design'),
  labels: AUDIO_EDITOR_LABELS,
  onChange: (file) => $('#clone-output-section').classList.toggle('is-complete', Boolean(file)),
})
const cloneProcessedOutput = new AudioEditor($('#clone-processed-output'), {
  label: t('finishing.processed', {}, 'Processed voice'),
  emptyTitle: t('output.emptyTitle', {}, 'Audio output'),
  emptyDescription: t('finishing.ready', {}, 'Ready for processing'),
  labels: AUDIO_EDITOR_LABELS,
  onChange: (file) => {
    if (file || !state.lastCloneGeneration?.processedBlob) return
    state.lastCloneGeneration.processedBlob = null
    state.lastCloneGeneration.postProcessing = null
    $('#processed-preview').hidden = true
    $('#post-processing-status').hidden = true
    $('[data-save-version="processed"]').disabled = true
    selectCloneVersion('original')
  },
})
const referenceAudio = new AudioEditor($('#reference-audio-preview'), {
  label: t('clone.referencePreview', {}, 'Reference preview'),
  emptyTitle: t('clone.noReference', {}, 'No reference selected'),
  emptyDescription: t('clone.chooseOrRecord', {}, 'Choose or record a sample'),
  labels: AUDIO_EDITOR_LABELS,
  onChange: (file) => {
    $('#reference-audio-drop').classList.toggle('has-file', Boolean(file))
    $('#reference-audio-name').textContent = file?.name || t('audio.noAudio', {}, 'No audio selected')
    $('#reference-audio-clear').hidden = !file
    if (file && state.activeTab === 'clone' && $('#voice-profile').value && !state.loadingProfile) renderVoiceProfileSelect('')
    updateWorkflowControls()
    updateDesignCompletion()
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
const gpuMonitor = new GpuMonitor($('#gpu-output'), { fetchJson })
const versionCheck = new VersionCheck($('#version-update'))
const dialogueScripts = new DialogueScriptLibrary({
  magicEditor,
  fetchJson,
  showToast,
  errorMessage,
  getInputType: () => state.inputType,
  setInputType,
  onDocumentChange: (documentText) => { state.inputDrafts['ssml-h'] = documentText },
})

for (const editor of [generateOutput, cloneOutput, cloneProcessedOutput, streamOutput, referenceAudio]) {
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

class WorkflowValidationError extends Error {
  constructor(message, targetSelector, focusSelector = null) {
    super(message)
    this.name = 'WorkflowValidationError'
    this.targetSelector = targetSelector
    this.focusSelector = focusSelector
  }
}

function emphasizeRequiredStep(targetSelector, message, focusSelector = null) {
  const target = $(targetSelector)
  if (target) {
    target.classList.remove('requires-attention')
    void target.offsetWidth
    target.classList.add('requires-attention')
    clearTimeout(target.attentionTimer)
    target.attentionTimer = setTimeout(() => target.classList.remove('requires-attention'), 1900)
    target.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' })
  }
  const focusTarget = focusSelector ? $(focusSelector) : null
  if (focusTarget && !focusTarget.disabled) setTimeout(() => focusTarget.focus({ preventScroll: true }), 180)
  showToast(message)
}

function presentWorkflowValidation(error) {
  if (!(error instanceof WorkflowValidationError)) return false
  emphasizeRequiredStep(error.targetSelector, error.message, error.focusSelector)
  setStatus(error.message, 'error')
  return true
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
      generationMode: state.generationMode,
      headerCollapsed: state.headerCollapsed,
      inputType: state.inputType,
      designSource: state.designSource,
      cloneMode: state.cloneMode,
    }))
  } catch {
    // Browser storage may be unavailable in privacy-restricted sessions.
  }
}

function readSessionJson(key) {
  try { return JSON.parse(sessionStorage.getItem(key) || 'null') } catch { return null }
}

function restoreSessionState() {
  const ui = readSessionJson(UI_SESSION_KEY)
  if (['generate', 'clone', 'api', 'system'].includes(ui?.activeTab)) {
    state.activeTab = ui.activeTab
  } else if (ui?.activeTab === 'stream') {
    state.activeTab = 'generate'
    state.generationMode = 'stream'
  }
  if (['generate', 'stream'].includes(ui?.generationMode)) state.generationMode = ui.generationMode
  if (typeof ui?.headerCollapsed === 'boolean') state.headerCollapsed = ui.headerCollapsed
  if (['magic', 'text', 'ssml', 'ssml-h'].includes(ui?.inputType)) state.inputType = ui.inputType
  if (['reference', 'direction'].includes(ui?.designSource)) state.designSource = ui.designSource
  if (['reference', 'transcript'].includes(ui?.cloneMode)) state.cloneMode = ui.cloneMode
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
  if (state.inputType === 'magic') {
    const stats = magicEditor.stats()
    $('#text-metrics').textContent = t('composer.metrics', { characters: stats.characters, words: stats.words }, `${stats.characters} characters · ${stats.words} words`)
    return
  }
  const text = $('#text-input').value
  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  $('#text-metrics').textContent = t('composer.metrics', { characters: text.length, words }, `${text.length} characters · ${words} words`)
}

function updateDesignMetrics() {
  const text = $('#design-text-input').value
  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  $('#design-text-metrics').textContent = t('composer.metrics', { characters: text.length, words }, `${text.length} characters · ${words} words`)
  updateDesignCompletion()
}

function loadDesignPortrait(file) {
  if (!file) return
  const allowed = ['image/png', 'image/jpeg', 'image/webp']
  if (!allowed.includes(file.type)) return showToast(t('errors.portraitFormat', {}, 'Choose a PNG, JPEG, or WebP portrait.'))
  if (file.size > 5 * 1024 * 1024) return showToast(t('errors.portraitSize', {}, 'Voice portraits must be 5 MB or smaller.'))
  setDesignPortrait({ file })
  return true
}

function inputSample(inputType) {
  if (inputType === 'magic') return SAMPLE_TEXTS[state.sampleIndex]
  if (inputType === 'text') return SAMPLE_TEXTS[state.sampleIndex]
  return INPUT_SAMPLES[inputType][0]
}

function setInputType(inputType, { syncSsmlH = true } = {}) {
  if (!['magic', 'text', 'ssml', 'ssml-h'].includes(inputType)) return
  const editor = $('#text-input')
  const currentType = state.inputType
  if (currentType !== 'magic') state.inputDrafts[currentType] = editor.value
  if (currentType === 'ssml-h' && inputType === 'magic' && syncSsmlH) {
    try {
      dialogueScripts.runWithoutDirty(() => magicEditor.loadSSMLH(editor.value))
    } catch (error) {
      showToast(errorMessage(error))
      return
    }
  }
  if (currentType === 'magic' && inputType === 'ssml-h') {
    state.inputDrafts['ssml-h'] = magicEditor.toSSMLH()
  }
  if (inputType !== 'magic') {
    if (state.inputDrafts[inputType] == null) {
      state.inputDrafts[inputType] = inputType === 'ssml-h' ? magicEditor.toSSMLH() : inputSample(inputType)
    }
    editor.value = state.inputDrafts[inputType]
    editor.dataset.inputType = inputType
    editor.spellcheck = inputType === 'text'
  }
  editor.hidden = inputType === 'magic'
  $('#magic-editor-shell').hidden = inputType !== 'magic'
  $('#composer-title').textContent = inputType === 'magic'
    ? t('magic.script', {}, 'Dialogue script')
    : t('composer.text', {}, 'Input text')
  state.inputType = inputType
  $$('.input-type-control button').forEach((button) => {
    const active = button.dataset.inputType === inputType
    button.classList.toggle('active', active)
    button.setAttribute('aria-pressed', String(active))
  })
  updateMetrics()
  persistUiState()
}

function handleMagicChange(editorInstance, detail = {}) {
  if (!detail.takesOnly) {
    state.inputDrafts['ssml-h'] = editorInstance.toSSMLH()
    dialogueScripts?.markDirty()
    updateMetrics()
  }
  if (detail.takesChanged || !detail.takesOnly) scheduleMagicAssembly()
}

function cancelMagicAssembly() {
  clearTimeout(state.magicAssemblyTimer)
  state.magicAssemblyTimer = null
  state.magicAssemblyRevision += 1
  state.magicAssemblyAbort?.abort()
  state.magicAssemblyAbort = null
}

function scheduleMagicAssembly() {
  cancelMagicAssembly()
  if (state.inputType !== 'magic' || state.generationMode !== 'generate' || state.magicGenerating) return
  if (!magicEditor.hasCompleteTakes()) {
    generateOutput.clear()
    generateFinisher.clear()
    return
  }
  const revision = state.magicAssemblyRevision
  state.magicAssemblyTimer = setTimeout(() => assembleMagicPreview(revision), 300)
}

async function assembleMagicPreview(revision) {
  if (revision !== state.magicAssemblyRevision || !magicEditor.hasCompleteTakes()) return
  const controller = new AbortController()
  state.magicAssemblyAbort = controller
  try {
    const result = await magicTakeStudio.assemble(magicEditor.blocks, {
      outputFormat: $('#output-format').value,
      normalizeLoudness: $('#normalize-loudness').checked,
      signal: controller.signal,
    })
    if (revision !== state.magicAssemblyRevision) return
    await generateOutput.load(result.blob, `voxcpmtts-magic.${result.extension}`)
    generateFinisher.setSource(result.blob, result.extension)
    setStatus(t('status.magicReassembled', {}, 'Dialogue preview updated'), 'success')
  } catch (error) {
    if (error.name !== 'AbortError') showToast(errorMessage(error))
  } finally {
    if (state.magicAssemblyAbort === controller) state.magicAssemblyAbort = null
  }
}

function setGenerationMode(mode, { persist = true } = {}) {
  if (!['generate', 'stream'].includes(mode) || state.streamAbort) return
  const modeChanged = state.generationMode !== mode
  state.generationMode = mode
  if (modeChanged) generateFinisher.clear()
  $$('.generation-mode-control button').forEach((button) => {
    const active = button.dataset.generationMode === mode
    button.classList.toggle('active', active)
    button.setAttribute('aria-pressed', String(active))
  })
  const streaming = mode === 'stream'
  const action = $('#generate-button')
  $('i', action).className = streaming ? 'icon-radio' : 'icon-audio-lines'
  $('span', action).textContent = streaming
    ? t('common.startStream', {}, 'Start stream')
    : t('common.generateAudio', {}, 'Generate audio')
  $('#stream-stop').hidden = !streaming
  $('#generate-result').hidden = streaming
  $('#stream-result').hidden = !streaming
  $('#generation-output-icon').className = streaming ? 'icon-radio' : 'icon-audio-lines'
  $('#generation-output-title').textContent = streaming
    ? t('output.stream', {}, 'Stream output')
    : t('output.generated', {}, 'Generated audio')
  $('#generation-output-badge').textContent = streaming
    ? t('output.live', {}, 'Live output')
    : t('output.final', {}, 'Final output')
  $('#timing-settings').hidden = streaming
  if (!streaming) streamWaveform.hide()
  refreshFormatOptions()
  updateTimestampState()
  if (persist) persistUiState()
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

function designVoiceTags() {
  return commaSeparatedTags($('#design-voice-tags').value)
}

function setDesignPortrait({ file = null, url = '', removed = false } = {}) {
  if (state.portraitObjectUrl) URL.revokeObjectURL(state.portraitObjectUrl)
  state.portraitFile = file
  state.portraitObjectUrl = file ? URL.createObjectURL(file) : null
  state.portraitRemoved = removed
  const preview = $('#design-portrait-preview')
  const source = state.portraitObjectUrl || url
  preview.src = source || ''
  preview.hidden = !source
  $('.portrait-placeholder').hidden = Boolean(source)
  $('#design-portrait-remove').hidden = !source
  $('#design-portrait-choose').classList.toggle('has-image', Boolean(source))
  renderVoiceSaveState()
}

function resetDesignIdentity({ preserveMetadata = false } = {}) {
  $('#design-voice-name').disabled = false
  if (!preserveMetadata) {
    $('#design-voice-name').value = ''
    $('#design-voice-tags').value = ''
    setDesignPortrait()
  }
  updateDesignCompletion()
}

function updateDesignCompletion() {
  const sampleReady = Boolean($('#design-text-input').value.trim())
  const normalizedName = normalizedProfileName($('#design-voice-name').value)
  const nameAvailable = Boolean(normalizedName) && (
    state.editingProfile?.id === normalizedName || !state.profiles.some((profile) => profile.id === normalizedName)
  )
  const hasReference = Boolean(referenceAudio.currentFile()) || selectedProfile()?.profile_type === 'cloned'
  const conditioningReady = state.cloneMode === 'transcript'
    ? Boolean($('#reference-text').value.trim())
    : Boolean($('#clone-control-input').value.trim())
  const voiceReady = state.designSource === 'direction'
    ? Boolean($('#clone-control-input').value.trim())
    : hasReference && (state.cloneMode === 'reference' || conditioningReady)

  $('.design-sample-field').classList.toggle('is-complete', sampleReady)
  $('#design-identity').classList.toggle('is-complete', nameAvailable)
  $('#reference-audio-drop').classList.toggle('is-complete', state.designSource === 'reference' && hasReference)
  $('#clone-direction-panel').classList.toggle('is-complete', !$('#clone-direction-panel').hidden && conditioningReady)
  $('#clone-transcript-panel').classList.toggle('is-complete', !$('#clone-transcript-panel').hidden && conditioningReady)
  $('#clone-panel').classList.toggle('is-complete', voiceReady)
}

function profileMetadataChanged() {
  const profile = state.editingProfile
  if (!profile) return false
  const currentTags = designVoiceTags()
  const savedTags = profile.tags || []
  const tagsChanged = currentTags.length !== savedTags.length || currentTags.some((tag, index) => tag !== savedTags[index])
  const portraitChanged = Boolean(state.portraitFile) || (state.portraitRemoved && profile.has_portrait)
  return tagsChanged || portraitChanged
}

function selectedProfile() {
  const id = $('#voice-profile').value
  return state.profiles.find((profile) => profile.id === id) || null
}

function updateVoiceProfileState() {
  const profile = selectedProfile()
  magicEditor.setDefaultVoice(profile)
  const note = $('#voice-profile-note')
  if (!profile) note.textContent = t('profiles.selectionHelp', {}, 'Use a saved design or clone sample')
  else if (profile.profile_type === 'cloned') note.textContent = t(
    profile.has_transcript ? 'profiles.clonedWithTranscript' : 'profiles.cloned',
    { name: profile.id },
    `${profile.id} · stored reference audio${profile.has_transcript ? ' · transcript' : ''}`,
  )
  else note.textContent = t('profiles.designed', { name: profile.id }, `${profile.id} · saved voice design`)
  updateWorkflowControls()
  renderCloneProfileList()
}

function renderVoiceProfileSelect(selected = $('#voice-profile').value) {
  populateSelect($('#voice-profile'), [
    { value: '', label: t('common.none', {}, 'None') },
    ...state.profiles.map((profile) => ({
      value: profile.id,
      label: `${profile.id} (${profile.profile_type === 'cloned' ? t('clone.reference', {}, 'reference') : t('clone.directionLower', {}, 'direction')})`,
    })),
  ], selected)
  magicEditor.setVoices(state.profiles)
  updateVoiceProfileState()
}

function restoreProfileGenerationSettings(profile) {
  const recipe = profile.recipe || {}
  if (profile.language && [...$('#language').options].some((option) => option.value === profile.language)) {
    $('#language').value = profile.language
  }
  if (Number.isFinite(recipe.cfg_value)) {
    $('#guidance').value = recipe.cfg_value
    $('#guidance-slider').value = recipe.cfg_value
  }
  if (Number.isInteger(recipe.inference_timesteps)) {
    $('#steps').value = recipe.inference_timesteps
    $('#steps-slider').value = recipe.inference_timesteps
  }
  if (Number.isInteger(recipe.seed)) $('#seed').value = recipe.seed
  if (typeof recipe.randomize_seed === 'boolean') $('#randomize-seed').checked = recipe.randomize_seed
  else if (Number.isInteger(recipe.seed)) $('#randomize-seed').checked = false
  if (typeof recipe.normalize === 'boolean') $('#normalize-text').checked = recipe.normalize
  if (typeof recipe.normalize_loudness === 'boolean') $('#normalize-loudness').checked = recipe.normalize_loudness
  if (typeof recipe.protect_long_audio === 'boolean') $('#protect-long-audio').checked = recipe.protect_long_audio
  if (typeof recipe.denoise === 'boolean') $('#denoise').checked = recipe.denoise
  if (recipe.output_format && [...$('#output-format').options].some((option) => option.value === recipe.output_format)) {
    $('#output-format').value = recipe.output_format
  }
  updateSeedState()
  generationToolbar.sync()
}

function restoreProfileRecipe(profile) {
  const recipe = profile.recipe || {}
  $('#design-text-input').value = recipe.sample_text || profile.ref_text || $('#design-text-input').value
  updateDesignMetrics()
  restoreProfileGenerationSettings(profile)
  restorePostProcessingRecipe(profile)
}

async function loadProfileReference(profile, { design = false } = {}) {
  const audioUrl = design ? (profile.design_audio_url || profile.audio_url) : profile.audio_url
  if (!audioUrl) {
    referenceAudio.clear()
    return
  }
  const response = await fetch(audioUrl, { cache: 'no-store' })
  if (!response.ok) throw new Error(await responseError(response))
  const blob = await response.blob()
  const contentType = blob.type || 'audio/wav'
  const extension = contentType.includes('mpeg') ? 'mp3' : contentType.split('/')[1]?.replace('x-', '') || 'wav'
  await referenceAudio.load(new File([blob], `${profile.id}.${extension}`, { type: contentType }), profile.id)
}

async function useProfile(profile, tab, { editing = false } = {}) {
  if (tab === 'clone') {
    if (editing) {
      state.editingProfile = profile
      state.lastCloneGeneration = null
      cloneOutput.clear()
      resetProcessedPreview({ hideFinishing: true })
      $('#design-voice-name').value = profile.id
      $('#design-voice-name').disabled = true
      $('#design-voice-tags').value = (profile.tags || []).join(', ')
      setDesignPortrait({ url: profile.portrait_url || '' })
    } else {
      cancelVoiceEdit()
    }
    state.loadingProfile = true
    try {
      const recipeSource = profile.recipe?.design_source
      const designSource = recipeSource || (profile.profile_type === 'designed' ? 'direction' : 'reference')
      setDesignSource(designSource)
      $('#clone-control-input').value = profile.control || ''
      $('#reference-text').value = Object.prototype.hasOwnProperty.call(profile.recipe || {}, 'reference_text')
        ? profile.recipe.reference_text
        : (profile.ref_text || '')
      setCloneMode(profile.recipe?.clone_mode || (profile.has_transcript ? 'transcript' : 'reference'))
      if (designSource === 'reference') await loadProfileReference(profile, { design: editing })
      else referenceAudio.clear()
      restoreProfileRecipe(profile)
      renderVoiceProfileSelect(profile.id)
      updateDesignCompletion()
    } finally {
      state.loadingProfile = false
    }
  } else {
    renderVoiceProfileSelect(profile.id)
    restoreProfileGenerationSettings(profile)
  }
  activateTab(tab)
  renderVoiceSaveState()
  setStatus(editing
    ? t('profiles.refining', { name: profile.id }, `Refining ${profile.id}`)
    : t('profiles.loaded', { name: profile.id }, `Voice ${profile.id} loaded`), 'success')
}

function renderCloneProfileList() {
  const list = $('#clone-profile-list')
  if (!list) return
  const query = $('#clone-profile-filter').value.trim().toLowerCase()
  const profiles = state.profiles
    .filter((profile) => [profile.id, profile.description, profile.profile_type, profile.control, ...(profile.tags || [])].join(' ').toLowerCase().includes(query))
    .sort((left, right) => {
      if (left.id === state.editingProfile?.id) return -1
      if (right.id === state.editingProfile?.id) return 1
      return left.id.localeCompare(right.id, browserLanguage())
    })
  $('#clone-profile-count').textContent = String(state.profiles.length)
  if (!profiles.length) {
    const empty = document.createElement('div')
    empty.className = 'clone-profile-empty'
    empty.textContent = state.profiles.length
      ? t('profiles.noMatches', {}, 'No saved voices match this search.')
      : t('profiles.none', {}, 'No saved voices yet.')
    list.replaceChildren(empty)
    return
  }
  const selected = $('#voice-profile').value
  list.replaceChildren(...profiles.map((profile) => {
    const card = document.createElement('article')
    card.className = 'clone-profile-card'
    card.classList.toggle('selected', profile.id === selected)
    card.classList.toggle('editing', profile.id === state.editingProfile?.id)
    const copy = document.createElement('div')
    copy.className = 'clone-profile-copy'
    const name = document.createElement('strong')
    name.textContent = profile.id
    if (profile.id === state.editingProfile?.id) {
      const editing = document.createElement('span')
      editing.className = 'editing-badge'
      editing.textContent = t('profiles.editing', {}, 'Editing')
      name.append(editing)
    }
    const description = document.createElement('span')
    description.textContent = profile.description || (profile.profile_type === 'designed'
      ? t('profiles.directionOnly', {}, 'Direction-only voice design')
      : t('profiles.storedReference', {}, 'Stored reference voice'))
    const metadata = document.createElement('div')
    metadata.className = 'clone-profile-metadata'
    ;[
      profile.profile_type === 'cloned' ? t('clone.referenceBadge', {}, 'Reference') : t('clone.direction', {}, 'Direction'),
      profile.has_transcript ? t('clone.transcript', {}, 'Transcript') : null,
      Number.isInteger(profile.recipe?.seed) ? t('profiles.seed', { seed: profile.recipe.seed }, `Seed ${profile.recipe.seed}`) : null,
      ...(profile.tags || []).map((tag) => `#${tag}`),
    ].filter(Boolean).forEach((label) => {
      const badge = document.createElement('span')
      badge.textContent = label
      metadata.append(badge)
    })
    copy.append(name, description, metadata)
    const portraitButton = document.createElement('button')
    portraitButton.type = 'button'
    portraitButton.className = 'clone-profile-portrait'
    portraitButton.title = profile.portrait_url
      ? t('profiles.replacePortrait', { name: profile.id }, `Replace ${profile.id} portrait`)
      : t('profiles.addPortrait', { name: profile.id }, `Add ${profile.id} portrait`)
    portraitButton.setAttribute('aria-label', portraitButton.title)
    if (profile.portrait_url) {
      const portrait = document.createElement('img')
      const version = encodeURIComponent(profile.created_at || 'current')
      portrait.src = `${profile.portrait_url}?v=${version}`
      portrait.alt = t('profiles.portraitAlt', { name: profile.id }, `${profile.id} portrait`)
      portrait.loading = 'lazy'
      portraitButton.append(portrait)
    } else {
      const placeholder = document.createElement('span')
      placeholder.className = 'portrait-placeholder'
      placeholder.textContent = '?'
      placeholder.setAttribute('aria-hidden', 'true')
      portraitButton.append(placeholder)
    }
    portraitButton.addEventListener('click', () => {
      state.portraitTargetProfile = profile
      $('#design-portrait-input').click()
    })
    card.append(portraitButton)
    if (profile.audio_url) {
      const audio = document.createElement('audio')
      audio.className = 'clone-profile-audio'
      audio.controls = true
      audio.preload = 'none'
      audio.src = profile.audio_url
      copy.append(audio)
    }
    const actions = document.createElement('div')
    actions.className = 'clone-profile-actions'
    const use = document.createElement('button')
    use.type = 'button'
    use.className = 'secondary-button'
    use.disabled = profile.id === selected
    use.innerHTML = profile.id === selected
      ? `<i class="icon-check"></i><span>${escapeHtml(t('profiles.selected', {}, 'Selected'))}</span>`
      : `<i class="icon-audio-lines"></i><span>${escapeHtml(t('profiles.use', {}, 'Use voice'))}</span>`
    use.addEventListener('click', () => useProfile(profile, 'clone').catch((error) => showToast(errorMessage(error))))
    const edit = document.createElement('button')
    edit.type = 'button'
    edit.className = 'icon-button bordered'
    edit.title = t('profiles.edit', { name: profile.id }, `Edit ${profile.id}`)
    edit.setAttribute('aria-label', edit.title)
    edit.innerHTML = '<i class="icon-sliders-horizontal"></i>'
    edit.addEventListener('click', () => useProfile(profile, 'clone', { editing: true }).catch((error) => showToast(errorMessage(error))))
    const remove = document.createElement('button')
    remove.type = 'button'
    remove.className = 'icon-button bordered danger-icon'
    remove.title = t('profiles.delete', { name: profile.id }, `Delete ${profile.id}`)
    remove.setAttribute('aria-label', remove.title)
    remove.innerHTML = '<i class="icon-x"></i>'
    remove.addEventListener('click', () => openDeleteProfileDialog(profile))
    actions.append(use, edit, remove)
    card.append(copy, actions)
    return card
  }))
}

async function refreshProfiles(selected) {
  const payload = await fetchJson('/tts/voice-profiles', { cache: 'no-store' })
  state.profiles = payload.data || []
  renderVoiceProfileSelect(selected)
  renderCloneProfileList()
}

function formatOptions(formats) {
  return Object.entries(formats)
    .filter(([, details]) => details.ui !== false)
    .map(([value, details]) => ({
      value,
      label: details.label || value.toUpperCase(),
    }))
}

function refreshFormatOptions() {
  const streaming = state.activeTab === 'generate' && state.generationMode === 'stream'
  const formats = streaming ? state.streamFormats : state.formats
  const preferred = $('#output-format').value || (streaming ? 'mp3' : 'mp3')
  populateSelect($('#output-format'), formatOptions(formats), preferred)
}

function setDesignSource(source, { persist = true } = {}) {
  if (!['reference', 'direction'].includes(source)) return
  state.designSource = source
  $$('.design-source-control button').forEach((button) => {
    const active = button.dataset.designSource === source
    button.classList.toggle('active', active)
    button.setAttribute('aria-pressed', String(active))
  })
  const usesReference = source === 'reference'
  $('[data-design-reference]').hidden = !usesReference
  $('.clone-mode-control').hidden = !usesReference
  $('#reference-audio').disabled = !usesReference
  if (usesReference) {
    setCloneMode(state.cloneMode, { persist: false })
  } else {
    $('#clone-direction-panel').hidden = false
    $('#clone-transcript-panel').hidden = true
    $('#clone-conditioning-copy').textContent = t('design.describe', {}, 'Describe the voice and delivery to create')
  }
  $('#clone-direction-label').innerHTML = usesReference
    ? `${escapeHtml(t('clone.cloneDirection', {}, 'Clone direction'))} <small>${escapeHtml(t('common.optional', {}, 'Optional'))}</small>`
    : `${escapeHtml(t('design.voiceDirection', {}, 'Voice direction'))} <small>${escapeHtml(t('common.required', {}, 'Required'))}</small>`
  updateWorkflowControls()
  updateDesignCompletion()
  if (persist) persistUiState()
}

function setCloneMode(mode, { persist = true } = {}) {
  if (!['reference', 'transcript'].includes(mode)) return
  state.cloneMode = mode
  $$('.clone-mode-control button').forEach((button) => {
    const active = button.dataset.cloneMode === mode
    button.classList.toggle('active', active)
    button.setAttribute('aria-pressed', String(active))
  })
  const usesReference = state.designSource === 'reference'
  $('#clone-direction-panel').hidden = usesReference && mode !== 'reference'
  $('#clone-transcript-panel').hidden = !usesReference || mode !== 'transcript'
  $('#clone-conditioning-copy').textContent = !usesReference
    ? t('design.describe', {}, 'Describe the voice and delivery to create')
    : mode === 'reference'
      ? t('clone.directionHelp', {}, 'Direct the delivery of the cloned voice')
      : t('clone.transcriptHelp', {}, 'Guide cloning with the exact reference words')
  updateDesignCompletion()
  if (persist) persistUiState()
}

function updateSeedState() {
  const randomized = $('#randomize-seed').checked
  $('#seed').disabled = randomized
  const lockButton = $('#design-seed-lock')
  lockButton.setAttribute('aria-pressed', String(!randomized))
  lockButton.title = randomized
    ? t('generation.lockSeed', {}, 'Lock generation seed')
    : t('generation.unlockSeed', {}, 'Unlock generation seed')
  $('i', lockButton).className = randomized ? 'icon-square' : 'icon-check'
  $('span', lockButton).textContent = randomized
    ? t('generation.seedUnlocked', {}, 'Seed unlocked')
    : t('generation.seedLocked', {}, 'Seed locked')
  lockButton.classList.toggle('active', !randomized)
  generationToolbar.sync()
}

function updateTimestampState() {
  const enabled = $('#generate-timestamps')
  $('#timestamp-level').disabled = enabled.disabled || !enabled.checked
}

function updateWorkflowControls() {
  const cloning = state.activeTab === 'clone'
  const profile = selectedProfile()
  $('#reference-audio').disabled = !cloning || state.designSource !== 'reference'
  $('#denoise').disabled = !cloning || state.designSource !== 'reference' || !state.status.load_denoiser
  $('#voice-profile-note').classList.toggle('active', Boolean(profile))
  updateDesignCompletion()
}

function buildPayload({ workflow = state.activeTab, streaming = false } = {}) {
  const cloning = workflow === 'clone'
  const usesReference = cloning && state.designSource === 'reference'
  const profileId = $('#voice-profile').value || null
  const outputFormat = streaming ? 'mp3' : $('#output-format').value
  const referenceText = usesReference && state.cloneMode === 'transcript'
    ? ($('#reference-text').value.trim() || null)
    : null
  const generatedText = !cloning && state.inputType === 'magic'
    ? magicEditor.toSSMLH()
    : $(cloning ? '#design-text-input' : '#text-input').value.trim()
  const payload = {
    text: generatedText,
    input_type: cloning ? 'text' : (state.inputType === 'magic' ? 'ssml-h' : state.inputType),
    language: $('#language').value || 'English',
    voice: usesReference ? 'reference' : 'auto',
    voice_profile: cloning ? (usesReference ? profileId : null) : profileId,
    clone_mode: usesReference ? state.cloneMode : 'auto',
    control: cloning
      ? (!usesReference || state.cloneMode === 'reference' ? ($('#clone-control-input').value.trim() || null) : null)
      : null,
    ref_text: referenceText,
    cfg_value: Number($('#guidance').value),
    inference_timesteps: Number($('#steps').value),
    normalize: !cloning && state.inputType !== 'text' ? false : $('#normalize-text').checked,
    normalize_loudness: $('#normalize-loudness').checked,
    protect_long_audio: $('#protect-long-audio').checked,
    denoise: usesReference && $('#denoise').checked,
    seed: Number($('#seed').value || 42),
    randomize_seed: $('#randomize-seed').checked,
    device: $('#device').value,
    output_format: outputFormat,
  }
  if (streaming) payload.stream_format = 'mp3'
  return payload
}

function prepareAudioRequest(options = {}) {
  const { workflow = state.activeTab, streaming = false, payloadOverride = null } = options
  const payload = payloadOverride || buildPayload({ workflow, streaming })
  const cloning = workflow === 'clone'
  const needsReference = cloning && payload.voice === 'reference'
  const reference = Object.prototype.hasOwnProperty.call(options, 'referenceOverride')
    ? options.referenceOverride
    : referenceAudio.currentFile()
  const profile = state.profiles.find((item) => item.id === payload.voice_profile) || null

  const hasText = cloning
    ? Boolean(payload.text)
    : state.inputType === 'magic' && !payloadOverride
      ? Boolean(magicEditor.plainText())
      : Boolean(payload.text)
  if (!hasText) {
    throw new WorkflowValidationError(
      t('errors.textRequired', {}, 'Enter text to synthesize.'),
      cloning ? '#design-composer' : '#composer',
      cloning ? '#design-text-input' : state.inputType === 'magic' ? '.magic-turn.active textarea, .magic-turn textarea' : '#text-input',
    )
  }
  if (needsReference && !reference && profile?.profile_type !== 'cloned') {
    throw new WorkflowValidationError(
      t('errors.referenceRequired', {}, 'Choose or record reference audio, or select a saved cloned voice.'),
      '#reference-audio-drop',
      '#reference-audio-choose',
    )
  }
  if (needsReference && payload.clone_mode === 'transcript' && !payload.ref_text && !profile?.has_transcript) {
    throw new WorkflowValidationError(
      t('errors.transcriptRequired', {}, 'Enter or transcribe the reference words for transcript-guided cloning.'),
      '#clone-transcript-panel',
      '#reference-text',
    )
  }
  if (cloning && !needsReference && !payload.control) {
    throw new WorkflowValidationError(
      t('errors.directionRequired', {}, 'Describe the voice you want to design.'),
      '#clone-direction-panel',
      '#clone-control-input',
    )
  }
  return { payload, reference }
}

async function requestAudioResponse(options = {}) {
  const { workflow = state.activeTab, streaming = false, signal } = options
  const { payload, reference } = prepareAudioRequest(options)
  const needsReference = workflow === 'clone' && payload.voice === 'reference'

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
  return { response, payload, reference }
}

async function requestAudio(options = {}) {
  const { response, payload, reference } = await requestAudioResponse(options)
  return {
    blob: await response.blob(),
    extension: response.headers.get('X-VoxCPM-Format') || payload.output_format,
    seed: response.headers.get('X-VoxCPM-Seed'),
    payload,
    reference,
  }
}

function magicPreviewPayload() {
  const payload = buildPayload({ workflow: 'generate' })
  return {
    ...payload,
    text: magicEditor.toSSMLH(undefined, { preview: true }),
    input_type: 'ssml-h',
    voice_profile: payload.voice_profile,
    output_format: 'wav',
    normalize: false,
  }
}

async function generateMagicPreview(block) {
  setGenerationBusy(true, 'generate')
  startActivityPolling('generate')
  setStatus(t('status.generatingTurn', {}, 'Generating turn preview'))
  try {
    const speechIndex = magicEditor.speechBlocks().findIndex((item) => item.id === block.id)
    if (speechIndex < 0) throw new Error(t('errors.turnMissing', {}, 'This speech turn is no longer in the dialogue.'))
    const result = await magicTakeStudio.take(magicPreviewPayload(), speechIndex)
    setStatus(t('status.turnReady', {}, 'Turn preview ready'), 'success')
    finishActivityPolling('generate', 'complete', t('status.turnReady', {}, 'Turn preview ready'))
    return result
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    finishActivityPolling('generate', 'failed', errorMessage(error))
    throw error
  } finally {
    setGenerationBusy(false, 'generate')
  }
}

function setMagicCharacterPortrait(file = null) {
  if (state.magicCharacterPortraitUrl) URL.revokeObjectURL(state.magicCharacterPortraitUrl)
  state.magicCharacterPortraitFile = file
  state.magicCharacterPortraitUrl = file ? URL.createObjectURL(file) : null
  const preview = $('#magic-character-portrait-preview')
  preview.src = state.magicCharacterPortraitUrl || ''
  preview.hidden = !state.magicCharacterPortraitUrl
  $('#magic-character-portrait-placeholder').hidden = Boolean(state.magicCharacterPortraitUrl)
}

function commaSeparatedTags(value) {
  const tags = value
    .split(',')
    .map((tag) => tag.trim().replace(/\s+/g, ' '))
    .filter(Boolean)
  return [...new Map(tags.map((tag) => [tag.toLowerCase(), tag])).values()].slice(0, 12)
}

async function openMagicCharacterDialog(block) {
  const preview = await magicEditor.ensurePreview(block.id)
  if (!preview) return
  state.magicCharacterBlockId = block.id
  setMagicCharacterPortrait()
  const speechTurns = magicEditor.blocks.filter((item) => item.type === 'speech')
  const turnNumber = Math.max(1, speechTurns.findIndex((item) => item.id === block.id) + 1)
  $('#magic-character-name').value = `character-${turnNumber}`
  $('#magic-character-tags').value = ($('#language').value || 'English').toLowerCase()
  $('#magic-character-description').value = ''
  $('#magic-character-dialog').showModal()
  $('#magic-character-name').focus()
  $('#magic-character-name').select()
}

function closeMagicCharacterDialog() {
  state.magicCharacterBlockId = null
  setMagicCharacterPortrait()
  $('#magic-character-dialog').close()
}

function magicCharacterRecipe(block, preview) {
  const payload = preview.payload
  const spokenText = magicEditor.spokenText(block).trim()
  return {
    design_source: 'reference',
    clone_mode: 'transcript',
    seed: Number(preview.seed ?? payload.seed ?? 42),
    randomize_seed: false,
    cfg_value: Number(payload.cfg_value),
    inference_timesteps: Number(payload.inference_timesteps),
    normalize: Boolean(payload.normalize),
    normalize_loudness: Boolean(payload.normalize_loudness),
    protect_long_audio: Boolean(payload.protect_long_audio),
    denoise: false,
    output_format: preview.extension || 'wav',
    sample_text: spokenText,
    reference_text: spokenText,
  }
}

function renderTimestamps(workflow, result) {
  const panel = $(`#${workflow}-timestamp-results`)
  const header = document.createElement('header')
  const title = document.createElement('strong')
  title.textContent = t('timestamps.resultTitle', { level: result.level }, `${result.level} timestamps`)
  const count = document.createElement('span')
  count.textContent = t('timestamps.aligned', { count: result.items.length }, `${result.items.length} aligned`)
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
  setStatus(t('status.aligning', {}, 'Aligning generated speech'))
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
    showToast(t('errors.timestampFailed', { error: errorMessage(error) }, `Audio is ready, but timestamp alignment failed: ${errorMessage(error)}`))
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
  const progressWorkflow = workflow === 'stream' ? 'generate' : workflow
  const panel = $(`#${progressWorkflow}-progress`)
  const phase = activity.phase || 'preparing'
  const index = ACTIVITY_STAGE_INDEX[phase] ?? 0
  panel.hidden = false
  panel.dataset.phase = phase
  const phaseTitles = {
    preparing: t('progress.preparing', {}, 'Preparing request'),
    loading_model: t('progress.loadingModel', {}, 'Loading model'),
    generating: t('progress.generating', {}, 'Generating speech'),
    streaming: t('progress.streaming', {}, 'Streaming speech'),
    encoding: t('progress.encoding', {}, 'Encoding audio'),
    complete: t('progress.complete', {}, 'Audio is ready'),
    failed: t('progress.failed', {}, 'Generation failed'),
  }
  $('.progress-copy strong', panel).textContent = phaseTitles[phase] || activity.message || phaseTitles.preparing
  $('.progress-copy > span', panel).textContent = phase === 'complete'
    ? phaseTitles.complete
    : phase === 'loading_model'
      ? t('progress.firstLoad', {}, 'First use can take longer while model weights enter GPU memory')
      : phase === 'encoding'
        ? t('progress.outputFormat', {
            format: workflow === 'stream' ? 'MP3' : ($('#output-format').value || 'audio').toUpperCase(),
          }, 'Preparing the selected output format')
        : t('progress.active', {}, 'The request is active')
  $$('[data-stage]', panel).forEach((dot, dotIndex) => {
    dot.classList.toggle('done', phase === 'complete' || dotIndex < index)
    dot.classList.toggle('active', phase !== 'complete' && phase !== 'failed' && dotIndex === index)
  })
}

function startActivityPolling(workflow) {
  clearInterval(state.activityTimer)
  renderInferenceProgress(workflow, { phase: 'preparing', message: t('progress.preparing', {}, 'Preparing request') })
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
  renderInferenceProgress(workflow, { phase, message })
  const panel = $(`#${workflow === 'stream' ? 'generate' : workflow}-progress`)
  clearTimeout(panel.hideTimer)
  panel.hideTimer = setTimeout(() => { panel.hidden = true }, phase === 'complete' ? 1800 : 5000)
}

function setGenerationBusy(active, workflow = 'generate') {
  const button = workflow === 'stream' ? $('#generate-button') : $(`#${workflow}-button`)
  button.disabled = active
  if (workflow === 'stream') $('#stream-stop').disabled = !active
  if (['generate', 'stream'].includes(workflow)) {
    $$('.generation-mode-control button').forEach((modeButton) => { modeButton.disabled = active })
    magicEditor.setBusy(active)
  }
}

async function generateMagicAudio() {
  let prepared
  try {
    prepared = prepareAudioRequest({ workflow: 'generate' })
  } catch (error) {
    if (!presentWorkflowValidation(error)) showToast(errorMessage(error))
    return
  }
  cancelMagicAssembly()
  state.magicGenerating = true
  generateFinisher.clear()
  setGenerationBusy(true, 'generate')
  startActivityPolling('generate')
  setStatus(t('status.generatingMagic', {}, 'Generating dialogue takes'))
  const payload = prepared.payload
  try {
    const result = await magicTakeStudio.render(payload, magicEditor.blocks)
    magicEditor.applyTakes(result.takes)
    await generateOutput.load(result.output.blob, `voxcpmtts-magic.${result.output.extension}`)
    generateFinisher.setSource(result.output.blob, result.output.extension)
    if (Number.isInteger(result.requestSeed)) {
      $('#last-generated-seed').value = result.requestSeed
      $('#seed').value = result.requestSeed
      generationToolbar.sync()
    }
    await alignGeneratedAudio('generate', result.output.blob, result.output.extension, payload)
    setStatus(t('status.complete', {}, 'Generation complete'), 'success')
    finishActivityPolling('generate', 'complete', t('progress.complete', {}, 'Audio is ready'))
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    if (!presentWorkflowValidation(error)) showToast(errorMessage(error))
    finishActivityPolling('generate', 'failed', errorMessage(error))
  } finally {
    state.magicGenerating = false
    setGenerationBusy(false, 'generate')
  }
}

async function generateAudio(workflow = 'generate') {
  if (workflow === 'generate' && state.inputType === 'magic') return generateMagicAudio()
  let prepared
  try {
    prepared = prepareAudioRequest({ workflow })
  } catch (error) {
    if (!presentWorkflowValidation(error)) showToast(errorMessage(error))
    return
  }
  const output = workflow === 'clone' ? cloneOutput : generateOutput
  if (workflow === 'generate') generateFinisher.clear()
  if (workflow === 'clone') {
    state.lastCloneGeneration = null
    resetProcessedPreview({ hideFinishing: true })
    renderVoiceSaveState()
  }
  setGenerationBusy(true, workflow)
  startActivityPolling(workflow)
  setStatus(t('status.generating', {}, 'Generating audio'))
  try {
    const { blob, extension, seed, payload, reference } = await requestAudio({
      workflow,
      payloadOverride: prepared.payload,
      referenceOverride: prepared.reference,
    })
    await output.load(blob, `voxcpmtts${workflow === 'clone' ? '-clone' : ''}.${extension}`)
    if (workflow === 'generate') generateFinisher.setSource(blob, extension)
    if (workflow === 'clone') {
      state.lastCloneGeneration = {
        blob,
        extension,
        seed,
        payload,
        reference,
        processedBlob: null,
        processedExtension: null,
        postProcessing: null,
        selectedVersion: 'original',
      }
      state.selectedCloneVersion = 'original'
      $('#voice-finishing').hidden = false
      selectCloneVersion('original')
      // Generated audio is a replacement candidate, not conditioning for the next preview.
      // Keep the loaded reference and transcript stable across repeated generations.
      renderVoiceSaveState()
      updateDesignCompletion()
    }
    if (seed !== null) {
      $('#last-generated-seed').value = seed
      $('#seed').value = seed
      generationToolbar.sync()
      if (workflow === 'generate' && state.inputType === 'magic') magicEditor.markFullGeneration(Number(seed))
    }
    await alignGeneratedAudio(workflow, blob, extension, payload)
    setStatus(t('status.complete', {}, 'Generation complete'), 'success')
    finishActivityPolling(workflow, 'complete', t('progress.complete', {}, 'Audio is ready'))
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    if (!presentWorkflowValidation(error)) showToast(errorMessage(error))
    finishActivityPolling(workflow, 'failed', errorMessage(error))
  } finally {
    setGenerationBusy(false, workflow)
  }
}

const streamWaveform = new StreamWaveform($('#stream-live-wave'), $('#stream-live'), $('#stream-live-state'), $('#stream-live-detail'))

async function streamAudio() {
  let prepared
  try {
    prepared = prepareAudioRequest({ workflow: 'stream', streaming: true })
  } catch (error) {
    if (!presentWorkflowValidation(error)) showToast(errorMessage(error))
    return
  }
  const controller = new AbortController()
  const chunks = []
  let playback = null
  let completedSeed = null
  state.streamAbort = controller
  generateFinisher.clear()
  setGenerationBusy(true, 'stream')
  startActivityPolling('stream')
  streamWaveform.surface.hidden = false
  streamWaveform.stateElement.textContent = t('stream.preparing', {}, 'Preparing stream')
  streamWaveform.detailElement.textContent = t('stream.buffered', { size: 0 }, '0 KiB buffered')
  setStatus(t('status.streaming', {}, 'Streaming audio'))
  try {
    playback = await IncrementalAudioPlayback.create(streamWaveform)
    state.streamPlayback = playback
    const { response } = await requestAudioResponse({
      workflow: 'stream',
      streaming: true,
      signal: controller.signal,
      payloadOverride: prepared.payload,
      referenceOverride: prepared.reference,
    })
    const seed = response.headers.get('X-VoxCPM-Seed')
    if (seed !== null) {
      $('#last-generated-seed').value = seed
      $('#seed').value = seed
      generationToolbar.sync()
      completedSeed = Number(seed)
    }
    if (!response.body) throw new Error(t('errors.streamUnavailable', {}, 'Streaming response body is unavailable in this browser.'))
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
      setStatus(t('status.streamingSize', { size: (totalBytes / 1024).toFixed(0) }, `Streaming audio · ${(totalBytes / 1024).toFixed(0)} KiB`))
    }
    let resumeAt = 0
    if (playback) {
      await playback.finish()
      resumeAt = playback.currentTime()
      playback.stop()
      playback = null
      state.streamPlayback = null
    }
    const streamedBlob = new Blob(chunks, { type: 'audio/mpeg' })
    await streamOutput.load(streamedBlob, 'voxcpmtts-stream.mp3')
    generateFinisher.setSource(streamedBlob, 'mp3')
    if (state.inputType === 'magic') magicEditor.markFullGeneration(completedSeed)
    if (resumeAt > 0) await streamOutput.playFrom(resumeAt).catch(() => {})
    streamWaveform.complete()
    setTimeout(() => streamWaveform.hide(), 1000)
    setStatus(t('status.streamComplete', {}, 'Stream complete'), 'success')
    finishActivityPolling('stream', 'complete', t('progress.complete', {}, 'Audio is ready'))
  } catch (error) {
    if (error.name === 'AbortError') {
      if (chunks.length) await streamOutput.load(new Blob(chunks, { type: 'audio/mpeg' }), 'voxcpmtts-stream-partial.mp3')
      streamWaveform.complete()
      setStatus(t('status.streamStopped', {}, 'Stream stopped'), 'success')
      finishActivityPolling('stream', 'complete', t('status.streamStopped', {}, 'Stream stopped'))
    }
    else {
      setStatus(errorMessage(error), 'error')
      if (!presentWorkflowValidation(error)) showToast(errorMessage(error))
      finishActivityPolling('stream', 'failed', errorMessage(error))
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
  if (!file) {
    return emphasizeRequiredStep(
      '#reference-audio-drop',
      t('errors.referenceFirst', {}, 'Choose or record reference audio first.'),
      '#reference-audio-choose',
    )
  }
  const button = $('#transcribe-reference')
  const label = button.querySelector('span')
  button.disabled = true
  const firstLoad = !state.status.asr_loaded
  label.textContent = firstLoad
    ? t('status.loadingTranscription', {}, 'Loading transcription model')
    : t('status.transcribing', {}, 'Transcribing reference')
  setStatus(label.textContent)
  try {
    const form = new FormData()
    form.append('reference_audio', file, file.name)
    form.append('language', 'auto')
    const result = await fetchJson('/tts/transcribe-upload', { method: 'POST', body: form })
    $('#reference-text').value = result.text || ''
    state.status.asr_loaded = true
    setCloneMode('transcript')
    setStatus(t('status.transcriptReady', {}, 'Reference transcript ready'), 'success')
  } catch (error) {
    setStatus(errorMessage(error), 'error')
    showToast(errorMessage(error))
  } finally {
    label.textContent = t('clone.transcribe', {}, 'Transcribe')
    button.disabled = !state.status.load_asr
  }
}

async function saveProfileRequest({
  name,
  profileType,
  description = '',
  tags = [],
  file = null,
  designFile = null,
  portraitFile = null,
  clearDesignAudio = false,
  clearPortrait = false,
  refText = '',
  control = '',
  recipe = null,
  editing = false,
  metadataOnly = false,
}) {
  const form = new FormData()
  if (!metadataOnly) {
    form.append('name', name)
    form.append('profile_type', profileType)
    form.append('description', description)
    form.append('ref_text', refText)
    form.append('control', control)
    form.append('language', $('#language').value || 'English')
    if (recipe) form.append('recipe', JSON.stringify(recipe))
    if (file) form.append('reference_audio', file, file.name)
    if (designFile) form.append('design_reference_audio', designFile, designFile.name)
    if (editing) form.append('clear_design_audio', String(clearDesignAudio))
  }
  form.append('tags', JSON.stringify(tags))
  if (portraitFile) form.append('portrait', portraitFile, portraitFile.name)
  if (editing) form.append('clear_portrait', String(clearPortrait))
  const path = editing ? `/tts/voice-profiles/${encodeURIComponent(name)}` : '/tts/voice-profiles'
  return fetchJson(path, { method: editing ? 'PUT' : 'POST', body: form })
}

function generatedVoiceRecipe(generated) {
  const payload = generated.payload
  const recipe = {
    design_source: payload.voice === 'reference' ? 'reference' : 'direction',
    clone_mode: payload.voice === 'reference' ? payload.clone_mode : 'reference',
    seed: Number(generated.seed ?? payload.seed ?? 42),
    randomize_seed: false,
    cfg_value: Number(payload.cfg_value),
    inference_timesteps: Number(payload.inference_timesteps),
    normalize: Boolean(payload.normalize),
    normalize_loudness: Boolean(payload.normalize_loudness),
    protect_long_audio: Boolean(payload.protect_long_audio),
    denoise: Boolean(payload.denoise),
    output_format: payload.output_format,
    sample_text: payload.text,
    reference_text: payload.ref_text || '',
  }
  if (generated.selectedVersion === 'processed' && generated.postProcessing) {
    recipe.post_processing = generated.postProcessing
  }
  return recipe
}

function signedDecibels(value) {
  const numeric = Number(value)
  const formatted = numeric.toFixed(1).replace(/\.0$/, '')
  return `${numeric > 0 ? '+' : ''}${formatted} dB`
}

function postProcessingOptions() {
  return {
    method: $('#post-method').value,
    preset: $('#post-preset').value,
    pitch_semitones: Number($('#post-pitch').value),
    speed_factor: Number($('#post-speed').value),
    noise_reduction_db: Number($('#post-noise').value),
    bass_db: Number($('#post-bass').value),
    presence_db: Number($('#post-presence').value),
    dynamics: Number($('#post-dynamics').value),
    normalize_loudness: $('#post-normalize').checked,
  }
}

function renderPostProcessingValues() {
  const pitch = Number($('#post-pitch').value)
  $('#post-pitch-value').textContent = `${pitch > 0 ? '+' : ''}${pitch.toFixed(1).replace(/\.0$/, '')} st`
  $('#post-speed-value').textContent = `${Number($('#post-speed').value).toFixed(2)}x`
  $('#post-noise-value').textContent = `${Number($('#post-noise').value).toFixed(1).replace(/\.0$/, '')} dB`
  $('#post-bass-value').textContent = signedDecibels($('#post-bass').value)
  $('#post-presence-value').textContent = signedDecibels($('#post-presence').value)
  $('#post-dynamics-value').textContent = `${$('#post-dynamics').value}%`
}

function postProcessingMethodLabel(method = $('#post-method').value) {
  return method === 'signalsmith'
    ? t('finishing.signalsmithVoice', {}, 'Signalsmith Voice')
    : t('finishing.ffmpegStudio', {}, 'FFmpeg Studio')
}

function renderPostProcessingMethod() {
  $('#signalsmith-controls').hidden = $('#post-method').value !== 'signalsmith'
  $('#processed-method-badge').textContent = postProcessingMethodLabel()
}

function selectCloneVersion(version) {
  const processedAvailable = Boolean(state.lastCloneGeneration?.processedBlob)
  state.selectedCloneVersion = version === 'processed' && processedAvailable ? 'processed' : 'original'
  if (state.lastCloneGeneration) state.lastCloneGeneration.selectedVersion = state.selectedCloneVersion
  $$('[data-save-version]').forEach((button) => {
    const active = button.dataset.saveVersion === state.selectedCloneVersion
    button.classList.toggle('active', active)
    button.setAttribute('aria-pressed', String(active))
  })
  renderVoiceSaveState()
}

function resetProcessedPreview({ hideFinishing = false } = {}) {
  if (state.lastCloneGeneration) {
    state.lastCloneGeneration.processedBlob = null
    state.lastCloneGeneration.postProcessing = null
  }
  cloneProcessedOutput.clear()
  $('#processed-preview').hidden = true
  $('#post-processing-status').hidden = true
  $('[data-save-version="processed"]').disabled = true
  selectCloneVersion('original')
  if (hideFinishing) $('#voice-finishing').hidden = true
}

function applyPostProcessingPreset(name, { invalidate = true } = {}) {
  const preset = POST_PROCESSING_PRESETS[name]
  if (!preset) return
  $('#post-pitch').value = preset.pitch_semitones
  $('#post-speed').value = preset.speed_factor
  $('#post-noise').value = preset.noise_reduction_db
  $('#post-bass').value = preset.bass_db
  $('#post-presence').value = preset.presence_db
  $('#post-dynamics').value = preset.dynamics
  $('#post-normalize').checked = preset.normalize_loudness
  renderPostProcessingValues()
  if (invalidate) resetProcessedPreview()
}

function restorePostProcessingRecipe(profile) {
  const recipe = profile.recipe?.post_processing
  if (!recipe) {
    $('#post-method').value = 'ffmpeg'
    $('#post-preset').value = 'studio'
    applyPostProcessingPreset('studio', { invalidate: false })
    renderPostProcessingMethod()
    return
  }
  $('#post-method').value = recipe.method || 'ffmpeg'
  $('#post-preset').value = recipe.preset || 'custom'
  $('#post-pitch').value = recipe.pitch_semitones ?? 0
  $('#post-speed').value = recipe.speed_factor ?? 1
  $('#post-noise').value = recipe.noise_reduction_db ?? 0
  $('#post-bass').value = recipe.bass_db ?? 0
  $('#post-presence').value = recipe.presence_db ?? 0
  $('#post-dynamics').value = recipe.dynamics ?? 0
  $('#post-normalize').checked = recipe.normalize_loudness ?? true
  renderPostProcessingMethod()
  renderPostProcessingValues()
}

async function requestPostProcessedAudio(blob, extension, options) {
  const form = new FormData()
  form.append('audio', blob, `voxcpmtts-source.${extension || 'wav'}`)
  form.append('options', JSON.stringify(options))
  const response = await fetch('/tts/postprocess-upload', { method: 'POST', body: form })
  if (!response.ok) throw new Error(await responseError(response))
  return response.blob()
}

async function processDesignedVoice() {
  const generated = state.lastCloneGeneration
  if (!generated) {
    return emphasizeRequiredStep(
      '#clone-output-section',
      t('errors.generateBeforeProcess', {}, 'Generate a voice before processing it.'),
      '#clone-button',
    )
  }
  const button = $('#post-process-voice')
  const status = $('#post-processing-status')
  const options = postProcessingOptions()
  button.disabled = true
  status.hidden = false
  status.dataset.tone = 'neutral'
  status.textContent = t('status.processingVoice', { method: postProcessingMethodLabel(options.method) }, 'Finishing voice with {method}')
  try {
    const blob = await requestPostProcessedAudio(generated.blob, generated.extension, options)
    generated.processedBlob = blob
    generated.processedExtension = 'wav'
    generated.postProcessing = options
    await cloneProcessedOutput.load(blob, 'voxcpmtts-processed.wav')
    $('#processed-preview').hidden = false
    $('[data-save-version="processed"]').disabled = false
    selectCloneVersion('processed')
    status.dataset.tone = 'success'
    status.textContent = t('status.processedVoiceReady', {}, 'Processed preview ready')
  } catch (error) {
    resetProcessedPreview()
    status.hidden = false
    status.dataset.tone = 'error'
    status.textContent = errorMessage(error)
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
}

function compactGeneratedReferenceText(value) {
  const normalized = value.replace(/\s+/g, ' ').trim()
  if (!normalized) return ''
  let sentences = []
  if (Intl.Segmenter) {
    const segmenter = new Intl.Segmenter(undefined, { granularity: 'sentence' })
    sentences = [...segmenter.segment(normalized)].map((item) => item.segment.trim()).filter(Boolean)
  } else {
    sentences = normalized.match(/[^.!?]+[.!?]+|[^.!?]+$/g)?.map((item) => item.trim()) || [normalized]
  }
  const selected = []
  for (const sentence of sentences.slice(0, 2)) {
    const candidate = [...selected, sentence].join(' ')
    if (candidate.length > MAX_GENERATED_REFERENCE_CHARACTERS) break
    selected.push(sentence)
  }
  if (selected.length) return selected.join(' ')
  const bounded = normalized.slice(0, MAX_GENERATED_REFERENCE_CHARACTERS + 1)
  const lastSpace = bounded.lastIndexOf(' ')
  return bounded.slice(0, lastSpace > 80 ? lastSpace : MAX_GENERATED_REFERENCE_CHARACTERS).trim()
}

async function prepareGeneratedVoiceReference() {
  const generated = state.lastCloneGeneration
  if (!generated) throw new Error(t('errors.generateBeforeStore', {}, 'Generate cloned audio before storing this voice.'))
  const referenceText = compactGeneratedReferenceText(generated.payload.text)
  if (!referenceText) throw new Error(t('errors.referenceTextMissing', {}, 'The generated input does not contain usable reference text.'))
  const originalText = generated.payload.text.replace(/\s+/g, ' ').trim()
  if (referenceText === originalText) {
    const useProcessed = generated.selectedVersion === 'processed' && generated.processedBlob
    const blob = useProcessed ? generated.processedBlob : generated.blob
    const extension = useProcessed ? generated.processedExtension : generated.extension
    return {
      file: new File(
        [blob],
        `voxcpmtts-voice-reference.${extension}`,
        { type: blob.type || 'application/octet-stream' },
      ),
      refText: referenceText,
    }
  }

  setStatus(t('status.compactReference', {}, 'Generating a compact voice reference'))
  const payload = {
    ...generated.payload,
    text: referenceText,
    input_type: 'text',
    output_format: 'wav',
    normalize_loudness: false,
    seed: Number(generated.seed ?? generated.payload.seed ?? 42),
    randomize_seed: false,
  }
  const compact = await requestAudio({
    workflow: 'clone',
    payloadOverride: payload,
    referenceOverride: generated.reference,
  })
  const useProcessed = generated.selectedVersion === 'processed' && generated.postProcessing
  const referenceBlob = useProcessed
    ? await requestPostProcessedAudio(compact.blob, compact.extension, generated.postProcessing)
    : compact.blob
  return {
    file: new File([referenceBlob], 'voxcpmtts-voice-reference.wav', { type: referenceBlob.type || 'audio/wav' }),
    refText: referenceText,
  }
}

function openQuickSaveDialog(profileType) {
  if (profileType === 'clone-generated' && !state.lastCloneGeneration) {
    return emphasizeRequiredStep(
      '#clone-output-section',
      t('errors.generateDesignedFirst', {}, 'Generate a designed voice before storing it.'),
      '#clone-button',
    )
  }
  const name = normalizedProfileName($('#design-voice-name').value)
  if (!name) {
    return emphasizeRequiredStep(
      '#design-identity',
      t('errors.nameVoice', {}, 'Name this voice before storing it.'),
      '#design-voice-name',
    )
  }
  if (state.profiles.some((profile) => profile.id === name)) return showToast(t('errors.voiceExists', { name }, `Voice ${name} already exists. Use Edit to refine it.`))
  state.quickSaveType = profileType
  $('#save-profile-title').textContent = t('profiles.storeNamed', { name }, `Store ${name}`)
  $('#save-profile-copy').textContent = t('profiles.storeCopy', {}, 'Add an optional description, then store the voice, portrait, tags, and complete design recipe.')
  $('#quick-profile-description').value = ''
  $('#save-profile-dialog').showModal()
  $('#quick-profile-description').focus()
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

function renderVoiceSaveState() {
  const row = $('#clone-store-row')
  const editing = state.editingProfile
  const hasGeneration = Boolean(state.lastCloneGeneration)
  const metadataChanged = profileMetadataChanged()
  row.hidden = !editing && !hasGeneration
  $('#clone-store-title').textContent = editing
    ? t('profiles.refineNamed', { name: editing.id }, `Refine ${editing.id}`)
    : t('profiles.keep', {}, 'Keep this voice')
  $('#clone-store-copy').textContent = editing
    ? (hasGeneration
        ? t('profiles.replaceRefined', { name: editing.id }, `Replace ${editing.id} with this refined voice`)
        : metadataChanged
          ? t('profiles.updateMetadataCopy', {}, 'Update tags or portrait without replacing the saved audio')
          : t('profiles.refineCopy', {}, 'Generate a refined voice or change its tags or portrait'))
    : t('profiles.storeRecipe', {}, 'Store a compact generated reference and its design recipe')
  $('#cancel-voice-edit').hidden = !editing
  const button = $('#store-generated-voice')
  $('span', button).textContent = editing && !hasGeneration
    ? t('profiles.updateDetails', {}, 'Update details')
    : editing
      ? t('profiles.updateVoice', {}, 'Update voice')
      : t('profiles.store', {}, 'Store voice')
  button.disabled = !hasGeneration && !metadataChanged
}

function cancelVoiceEdit() {
  state.editingProfile = null
  resetDesignIdentity()
  renderCloneProfileList()
  renderVoiceSaveState()
}

function openUpdateProfileDialog() {
  if (!state.editingProfile || (!state.lastCloneGeneration && !profileMetadataChanged())) return
  const replacingVoice = Boolean(state.lastCloneGeneration)
  $('#update-profile-title').textContent = replacingVoice
    ? t('profiles.replaceTitle', {}, 'Replace saved voice?')
    : t('profiles.updateTitle', {}, 'Update saved voice?')
  const copy = $('#update-profile-copy')
  const name = document.createElement('strong')
  name.id = 'update-profile-name'
  name.textContent = state.editingProfile.id
  copy.replaceChildren(
    replacingVoice ? t('profiles.replacePrefix', {}, 'Replace ') : t('profiles.updatePrefix', {}, 'Update tags or portrait for '),
    name,
    replacingVoice ? t('profiles.replaceSuffix', {}, ' with this newly refined voice?') : t('profiles.updateSuffix', {}, ' without replacing its audio or design recipe?'),
  )
  $('span', $('#update-profile-confirm')).textContent = replacingVoice
    ? t('profiles.updateVoice', {}, 'Update voice')
    : t('profiles.updateDetails', {}, 'Update details')
  $('#update-profile-dialog').showModal()
}

function closeUpdateProfileDialog() {
  if ($('#update-profile-dialog').open) $('#update-profile-dialog').close()
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

async function refreshSystemStatus() {
  try {
    const status = await fetchJson('/tts/status')
    $('#readiness-output').textContent = JSON.stringify(status, null, 2)
  } catch (error) {
    $('#readiness-output').textContent = errorMessage(error)
  }
}

async function refreshSystem() {
  await Promise.all([refreshSystemStatus(), gpuMonitor.refresh()])
}

function activateTab(tab) {
  state.activeTab = tab
  const workflowActive = ['generate', 'clone'].includes(tab)
  $('.workspace').dataset.view = tab
  $$('.tab-button').forEach((button) => {
    const active = button.dataset.tab === tab
    button.classList.toggle('active', active)
    button.setAttribute('aria-selected', String(active))
  })
  $$('.tab-panel').forEach((panel) => { panel.hidden = panel.dataset.panel !== tab })
  $('#inference-settings').hidden = !workflowActive
  $('#composer').hidden = tab !== 'generate'
  $('#generate-authoring').hidden = tab !== 'generate'
  updateWorkflowControls()
  if (tab !== 'clone') referenceRecorder.stop()
  else requestAnimationFrame(() => referenceRecorder.refresh())
  gpuMonitor.stop()
  if (tab === 'api') refreshApi().catch((error) => showToast(errorMessage(error)))
  if (tab === 'system') {
    refreshSystemStatus()
    gpuMonitor.start()
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
  button.setAttribute('aria-label', collapsed ? t('hero.expand', {}, 'Expand header') : t('hero.collapse', {}, 'Collapse header'))
  button.title = collapsed ? t('hero.expand', {}, 'Expand header') : t('hero.collapse', {}, 'Collapse header')
  button.innerHTML = `<i class="icon-chevron-${collapsed ? 'down' : 'up'}"></i>`
  persistUiState()
}

function bindRangeInputs() {
  $$('[data-value-input]').forEach((range) => {
    const number = $(`#${range.dataset.valueInput}`)
    range.addEventListener('input', () => {
      number.value = range.value
      generationToolbar.sync()
    })
  })
  $$('[data-range-input]').forEach((number) => {
    const range = $(`#${number.dataset.rangeInput}`)
    number.addEventListener('input', () => {
      range.value = number.value
      generationToolbar.sync()
    })
  })
}

function resetControls() {
  $('#guidance').value = 2
  $('#guidance-slider').value = 2
  $('#steps').value = state.defaults.inference_timesteps || 10
  $('#steps-slider').value = state.defaults.inference_timesteps || 10
  $('#normalize-text').checked = false
  $('#normalize-loudness').checked = state.defaults.normalize_loudness ?? true
  $('#protect-long-audio').checked = state.defaults.protect_long_audio ?? true
  $('#denoise').checked = false
  $('#seed').value = state.defaults.seed ?? 42
  $('#randomize-seed').checked = state.defaults.randomize_seed ?? true
  updateSeedState()
  generationToolbar.sync()
}

async function initialize() {
  const [defaults, status, languages, formats, streamFormats, profiles, scripts, ssmlCapabilities] = await Promise.all([
    fetchJson('/tts/defaults'),
    fetchJson('/tts/status'),
    fetchJson('/tts/languages'),
    fetchJson('/tts/formats'),
    fetchJson('/tts/stream-formats'),
    fetchJson('/tts/voice-profiles'),
    fetchJson('/tts/dialogue-scripts'),
    fetchJson('/tts/ssml/capabilities'),
  ])
  state.defaults = defaults
  state.status = status
  state.formats = formats.formats || {}
  state.streamFormats = streamFormats.formats || {}
  state.profiles = profiles.data || []
  state.ssmlCapabilities = ssmlCapabilities

  populateSelect($('#language'), languages.languages.map((language) => ({ value: language, label: languageLabel(language) })), defaults.language)
  magicEditor.setLanguages(languages.languages.map((language) => ({ value: language, label: languageLabel(language) })))
  populateSelect($('#device'), status.hardware || [{ value: 'auto', label: t('common.auto', {}, 'Auto') }, { value: 'cpu', label: 'CPU' }], defaults.device)
  renderVoiceProfileSelect()
  renderCloneProfileList()
  dialogueScripts.initialize(scripts.data || [])
  magicEditor.setCapabilities(ssmlCapabilities)
  resetControls()
  $('#denoise').disabled = !status.load_denoiser
  $('#transcribe-reference').disabled = !status.load_asr
  const timestampsAvailable = Boolean(status.timestamps?.available)
  $('#generate-timestamps').disabled = !timestampsAvailable
  $('#timestamp-state').textContent = timestampsAvailable ? t('timestamps.ready', {}, 'Ready') : t('timestamps.unavailable', {}, 'Unavailable')
  $('#timestamp-state').dataset.state = timestampsAvailable ? 'available' : 'unavailable'
  updateTimestampState()
  $('#runtime-badge').dataset.state = 'ready'
  $('#runtime-state').textContent = t('runtime.backendReady', { backend: status.backend === 'nano' ? 'Nano' : t('runtime.native', {}, 'Native') }, `${status.backend === 'nano' ? 'Nano' : 'Native'} backend ready`)
  $('#runtime-model').textContent = `${status.model_id} · ${status.runtime}`
  versionCheck.check(status)
  setStatus(t('status.ready', {}, 'Ready'), 'success')
  setDesignSource(state.designSource, { persist: false })
  setCloneMode(state.cloneMode, { persist: false })
  updateDesignMetrics()
  refreshFormatOptions()

  setHeaderCollapsed(state.headerCollapsed)
  setGenerationMode(state.generationMode, { persist: false })
  activateTab(state.activeTab)
}

$('#text-input').addEventListener('input', () => {
  if (state.inputType !== 'magic') state.inputDrafts[state.inputType] = $('#text-input').value
  if (state.inputType === 'ssml-h') dialogueScripts.markDirty()
  updateMetrics()
})
$('#design-text-input').addEventListener('input', updateDesignMetrics)
$('#design-voice-name').addEventListener('input', updateDesignCompletion)
$('#design-voice-tags').addEventListener('input', renderVoiceSaveState)
$('#clone-control-input').addEventListener('input', updateDesignCompletion)
$('#reference-text').addEventListener('input', updateDesignCompletion)
$$('.input-type-control button').forEach((button) => button.addEventListener('click', () => setInputType(button.dataset.inputType)))
$$('.generation-mode-control button').forEach((button) => button.addEventListener('click', () => setGenerationMode(button.dataset.generationMode)))
$$('.design-source-control button').forEach((button) => button.addEventListener('click', () => setDesignSource(button.dataset.designSource)))
$$('.clone-mode-control button').forEach((button) => button.addEventListener('click', () => setCloneMode(button.dataset.cloneMode)))
$('#voice-profile').addEventListener('change', () => {
  const profile = selectedProfile()
  if (profile) restoreProfileGenerationSettings(profile)
  updateVoiceProfileState()
})
$('#randomize-seed').addEventListener('change', updateSeedState)
$('#design-seed-lock').addEventListener('click', () => {
  $('#randomize-seed').checked = !$('#randomize-seed').checked
  updateSeedState()
})
$('#generate-timestamps').addEventListener('change', updateTimestampState)
$('#sample-button').addEventListener('click', () => {
  if (state.inputType === 'magic' || state.inputType === 'text') state.sampleIndex = (state.sampleIndex + 1) % SAMPLE_TEXTS.length
  if (state.inputType === 'magic') {
    magicEditor.loadSample(inputSample('magic'))
    return
  }
  $('#text-input').value = inputSample(state.inputType)
  state.inputDrafts[state.inputType] = $('#text-input').value
  updateMetrics()
})
$('#design-sample-button').addEventListener('click', () => {
  state.sampleIndex = (state.sampleIndex + 1) % SAMPLE_TEXTS.length
  $('#design-text-input').value = SAMPLE_TEXTS[state.sampleIndex]
  updateDesignMetrics()
})
$('#design-portrait-choose').addEventListener('click', () => {
  state.portraitTargetProfile = null
  $('#design-portrait-input').click()
})
$('#design-portrait-input').addEventListener('change', async (event) => {
  const file = event.target.files[0]
  const targetProfile = state.portraitTargetProfile
  state.portraitTargetProfile = null
  event.target.value = ''
  if (!file) return
  if (!targetProfile) {
    loadDesignPortrait(file)
    return
  }
  const allowed = ['image/png', 'image/jpeg', 'image/webp']
  if (!allowed.includes(file.type)) return showToast(t('errors.portraitFormat', {}, 'Choose a PNG, JPEG, or WebP portrait.'))
  if (file.size > 5 * 1024 * 1024) return showToast(t('errors.portraitSize', {}, 'Voice portraits must be 5 MB or smaller.'))
  try {
    setStatus(t('status.updatingPortrait', { name: targetProfile.id }, `Updating ${targetProfile.id} portrait`))
    const saved = await saveProfileRequest({
      name: targetProfile.id,
      profileType: targetProfile.profile_type,
      tags: state.editingProfile?.id === targetProfile.id ? designVoiceTags() : (targetProfile.tags || []),
      portraitFile: file,
      editing: true,
      metadataOnly: true,
    })
    await refreshProfiles(saved.id)
    if (state.editingProfile?.id === saved.id) {
      state.editingProfile = state.profiles.find((profile) => profile.id === saved.id) || saved
      setDesignPortrait({ url: state.editingProfile.portrait_url || '' })
    }
    renderCloneProfileList()
    showToast(t('profiles.portraitUpdated', { name: saved.id }, `Updated ${saved.id} portrait.`), 'success')
    setStatus(t('status.portraitUpdated', { name: saved.id }, `Voice ${saved.id} portrait updated`), 'success')
  } catch (error) {
    showToast(errorMessage(error))
  }
})
$('#design-portrait-remove').addEventListener('click', (event) => {
  event.stopPropagation()
  setDesignPortrait({ removed: true })
})
$('#reference-audio').addEventListener('change', (event) => {
  const file = event.target.files[0]
  if (file) referenceAudio.load(file, file.name)
  event.target.value = ''
})
$('#reference-audio-choose').addEventListener('click', () => $('#reference-audio').click())
$('#reference-audio-clear').addEventListener('click', () => referenceAudio.clear())
$('#reference-audio-preview').addEventListener('click', (event) => {
  if (event.target.closest('[data-role="empty"]')) $('#reference-audio').click()
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
$('#clone-profile-filter').addEventListener('input', renderCloneProfileList)
$('#cancel-voice-edit').addEventListener('click', cancelVoiceEdit)
$('#post-method').addEventListener('change', () => {
  renderPostProcessingMethod()
  resetProcessedPreview()
})
$('#post-preset').addEventListener('change', (event) => {
  if (event.target.value === 'custom') return resetProcessedPreview()
  applyPostProcessingPreset(event.target.value)
})
for (const control of $$('#post-pitch, #post-speed, #post-noise, #post-bass, #post-presence, #post-dynamics')) {
  control.addEventListener('input', () => {
    $('#post-preset').value = 'custom'
    renderPostProcessingValues()
    resetProcessedPreview()
  })
}
$('#post-normalize').addEventListener('change', () => {
  $('#post-preset').value = 'custom'
  resetProcessedPreview()
})
$('#post-process-voice').addEventListener('click', processDesignedVoice)
$$('[data-save-version]').forEach((button) => {
  button.addEventListener('click', () => selectCloneVersion(button.dataset.saveVersion))
})
$('#store-generated-voice').addEventListener('click', () => {
  if (state.editingProfile) openUpdateProfileDialog()
  else openQuickSaveDialog('clone-generated')
})
$('#quick-save-form').addEventListener('submit', async (event) => {
  event.preventDefault()
  const profileType = state.quickSaveType
  const name = normalizedProfileName($('#design-voice-name').value)
  if (!profileType || !name) return showToast(t('errors.nameVoice', {}, 'Name this voice before storing it.'))
  const button = $('button[type="submit"]', event.currentTarget)
  button.disabled = true
  try {
    const generatedReference = profileType === 'clone-generated'
      ? await prepareGeneratedVoiceReference()
      : null
    const saved = await saveProfileRequest({
      name,
      profileType: profileType === 'clone-generated' ? 'cloned' : profileType,
      description: $('#quick-profile-description').value.trim(),
      tags: designVoiceTags(),
      file: generatedReference?.file || null,
      designFile: state.lastCloneGeneration?.payload.voice === 'reference' ? state.lastCloneGeneration.reference : null,
      portraitFile: state.portraitFile,
      refText: generatedReference?.refText || '',
      control: state.lastCloneGeneration?.payload.control || '',
      recipe: generatedVoiceRecipe(state.lastCloneGeneration),
    })
    closeQuickSaveDialog()
    await refreshProfiles(saved.id)
    state.editingProfile = state.profiles.find((profile) => profile.id === saved.id) || saved
    state.lastCloneGeneration = null
    resetProcessedPreview({ hideFinishing: true })
    $('#design-voice-name').value = saved.id
    $('#design-voice-name').disabled = true
    setDesignPortrait({ url: state.editingProfile.portrait_url || '' })
    renderVoiceSaveState()
    renderCloneProfileList()
    showToast(t('profiles.savedNamed', { name: saved.id }, `Saved ${saved.id}.`), 'success')
    setStatus(t('status.voiceStored', { name: saved.id }, `Voice ${saved.id} stored`), 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})
$('#save-profile-close').addEventListener('click', closeQuickSaveDialog)
$('#save-profile-cancel').addEventListener('click', closeQuickSaveDialog)
$('#save-profile-dialog').addEventListener('click', (event) => { if (event.target === event.currentTarget) closeQuickSaveDialog() })
$('#magic-character-portrait').addEventListener('change', (event) => {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  const allowed = ['image/png', 'image/jpeg', 'image/webp']
  if (!allowed.includes(file.type)) return showToast(t('errors.portraitFormat', {}, 'Choose a PNG, JPEG, or WebP portrait.'))
  if (file.size > 5 * 1024 * 1024) return showToast(t('errors.portraitSize', {}, 'Voice portraits must be 5 MB or smaller.'))
  setMagicCharacterPortrait(file)
})
$('#magic-character-form').addEventListener('submit', async (event) => {
  event.preventDefault()
  const blockId = state.magicCharacterBlockId
  const block = magicEditor.block(blockId)
  const name = normalizedProfileName($('#magic-character-name').value)
  if (!block || !name) return showToast(t('errors.nameVoice', {}, 'Name this voice before storing it.'))
  if (state.profiles.some((profile) => profile.id === name)) {
    return showToast(t('errors.voiceExists', { name }, `Voice ${name} already exists. Use Edit to refine it.`))
  }
  const button = $('button[type="submit"]', event.currentTarget)
  button.disabled = true
  try {
    const preview = await magicEditor.ensurePreview(block.id)
    if (!preview) throw new Error(t('errors.generateTurnFirst', {}, 'Generate this turn before saving its voice.'))
    const extension = preview.extension || 'wav'
    const file = new File([preview.blob], `${name}.${extension}`, { type: preview.blob.type || 'audio/wav' })
    const selected = $('#voice-profile').value
    const saved = await saveProfileRequest({
      name,
      profileType: 'cloned',
      description: $('#magic-character-description').value.trim(),
      tags: commaSeparatedTags($('#magic-character-tags').value),
      file,
      designFile: file,
      portraitFile: state.magicCharacterPortraitFile,
      refText: magicEditor.spokenText(block).trim(),
      control: block.direction || '',
      recipe: magicCharacterRecipe(block, preview),
    })
    await refreshProfiles(selected)
    magicEditor.assignVoice(block.id, saved.id)
    closeMagicCharacterDialog()
    showToast(t('magic.characterSaved', { name: saved.id }, `Saved ${saved.id} and assigned it to this turn.`), 'success')
    setStatus(t('status.voiceStored', { name: saved.id }, `Voice ${saved.id} stored`), 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})
$('#magic-character-close').addEventListener('click', closeMagicCharacterDialog)
$('#magic-character-cancel').addEventListener('click', closeMagicCharacterDialog)
$('#magic-character-dialog').addEventListener('click', (event) => { if (event.target === event.currentTarget) closeMagicCharacterDialog() })
$('#magic-character-dialog').addEventListener('cancel', (event) => {
  event.preventDefault()
  closeMagicCharacterDialog()
})
$('#update-profile-close').addEventListener('click', closeUpdateProfileDialog)
$('#update-profile-cancel').addEventListener('click', closeUpdateProfileDialog)
$('#update-profile-dialog').addEventListener('click', (event) => { if (event.target === event.currentTarget) closeUpdateProfileDialog() })
$('#update-profile-confirm').addEventListener('click', async (event) => {
  const profile = state.editingProfile
  const generated = state.lastCloneGeneration
  if (!profile || (!generated && !profileMetadataChanged())) return
  const button = event.currentTarget
  button.disabled = true
  try {
    let saved
    if (generated) {
      const generatedReference = await prepareGeneratedVoiceReference()
      saved = await saveProfileRequest({
        name: profile.id,
        profileType: 'cloned',
        description: profile.description || '',
        tags: designVoiceTags(),
        file: generatedReference.file,
        designFile: generated.payload.voice === 'reference' ? generated.reference : null,
        portraitFile: state.portraitFile,
        clearDesignAudio: generated.payload.voice !== 'reference',
        clearPortrait: state.portraitRemoved,
        refText: generatedReference.refText,
        control: generated.payload.control || '',
        recipe: generatedVoiceRecipe(generated),
        editing: true,
      })
    } else {
      saved = await saveProfileRequest({
        name: profile.id,
        profileType: profile.profile_type,
        tags: designVoiceTags(),
        portraitFile: state.portraitFile,
        clearPortrait: state.portraitRemoved,
        editing: true,
        metadataOnly: true,
      })
    }
    closeUpdateProfileDialog()
    state.lastCloneGeneration = null
    resetProcessedPreview({ hideFinishing: true })
    await refreshProfiles(saved.id)
    state.editingProfile = state.profiles.find((item) => item.id === saved.id) || saved
    setDesignPortrait({ url: state.editingProfile.portrait_url || '' })
    renderVoiceSaveState()
    renderCloneProfileList()
    showToast(generated
      ? t('profiles.updatedNamed', { name: saved.id }, `Updated ${saved.id}.`)
      : t('profiles.detailsUpdatedNamed', { name: saved.id }, `Updated ${saved.id} details.`), 'success')
    setStatus(generated
      ? t('status.voiceUpdated', { name: saved.id }, `Voice ${saved.id} updated`)
      : t('status.voiceDetailsUpdated', { name: saved.id }, `Voice ${saved.id} details updated`), 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})
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
    if (state.editingProfile?.id === profile.id) cancelVoiceEdit()
    await refreshProfiles()
    showToast(t('profiles.deletedNamed', { name: profile.id }, `Deleted ${profile.id}.`), 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})
$('#generate-button').addEventListener('click', () => {
  if (state.generationMode === 'stream') streamAudio()
  else generateAudio('generate')
})
$('#clone-button').addEventListener('click', () => generateAudio('clone'))
$('#stream-stop').addEventListener('click', () => {
  state.streamAbort?.abort()
  state.streamPlayback?.stop()
})
$('#transcribe-reference').addEventListener('click', transcribeReference)
$('#reset-controls').addEventListener('click', resetControls)
$$('[data-generation-reset]').forEach((button) => button.addEventListener('click', resetControls))
$('#api-refresh').addEventListener('click', () => refreshApi().catch((error) => showToast(errorMessage(error))))
$('#system-refresh').addEventListener('click', refreshSystem)
$('#purge-models').addEventListener('click', async () => {
  try {
    const result = await fetchJson('/tts/purge', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
    showToast(t('system.purged', { count: result.purged.length }, `Purged ${result.purged.length} model cache entries.`), 'success')
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
  $('#runtime-state').textContent = t('runtime.serviceUnavailable', {}, 'Service unavailable')
  $('#runtime-model').textContent = errorMessage(error)
  setStatus(errorMessage(error), 'error')
})

document.addEventListener('visibilitychange', () => {
  if (document.hidden) gpuMonitor.stop()
  else if (state.activeTab === 'system') gpuMonitor.start()
})
window.addEventListener('beforeunload', () => {
  gpuMonitor.stop()
  state.streamAbort?.abort()
  state.streamPlayback?.stop()
  clearInterval(state.activityTimer)
  streamWaveform.hide()
  referenceRecorder.stop()
  generateOutput.destroy()
  generateFinisher.destroy()
  cloneOutput.destroy()
  cloneProcessedOutput.destroy()
  streamOutput.destroy()
  referenceAudio.destroy()
})
