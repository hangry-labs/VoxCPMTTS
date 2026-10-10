import { AudioEditor } from './audio-editor.js?v=waveform-hitbox'

const PRESETS = {
  clean: { pitch_semitones: 0, speed_factor: 1, noise_reduction_db: 1, bass_db: 0, presence_db: 0.5, dynamics: 15, normalize_loudness: true },
  studio: { pitch_semitones: 0, speed_factor: 1, noise_reduction_db: 2, bass_db: 1, presence_db: 1, dynamics: 35, normalize_loudness: true },
}

function signedDecibels(value) {
  const numeric = Number(value)
  const formatted = numeric.toFixed(1).replace(/\.0$/, '')
  return `${numeric > 0 ? '+' : ''}${formatted} dB`
}

export class AudioFinisher {
  constructor(root, { audioEditorLabels, labels, responseError, onError }) {
    this.root = root
    this.labels = labels
    this.responseError = responseError
    this.onError = onError
    this.source = null
    this.processedOptions = null
    this.$ = (role) => root.querySelector(`[data-finish-role="${role}"]`)
    this.output = new AudioEditor(this.$('output'), {
      label: labels.processed,
      emptyTitle: labels.emptyTitle,
      emptyDescription: labels.ready,
      labels: audioEditorLabels,
      onChange: (file) => {
        if (!file) {
          this.processedOptions = null
          this.$('preview').hidden = true
          this.renderStale(false)
        }
      },
    })
    this.output.container.addEventListener('audio-error', (event) => this.onError(event.detail))
    this.bind()
    this.applyPreset('studio', { invalidate: false })
    this.renderMethod()
  }

  bind() {
    this.adjustments = this.root.querySelector('.audio-studio-adjustments')
    this.handleDocumentPointerDown = (event) => {
      if (this.adjustments?.open && !this.adjustments.contains(event.target)) this.adjustments.open = false
    }
    this.handleDocumentKeyDown = (event) => {
      if (event.key === 'Escape' && this.adjustments?.open) {
        this.adjustments.open = false
        this.adjustments.querySelector('summary')?.focus()
      }
    }
    document.addEventListener('pointerdown', this.handleDocumentPointerDown)
    document.addEventListener('keydown', this.handleDocumentKeyDown)
    this.$('method').addEventListener('change', () => {
      this.renderMethod()
      this.markPreviewStale()
    })
    this.$('preset').addEventListener('change', (event) => {
      if (event.target.value === 'custom') return this.markPreviewStale()
      this.applyPreset(event.target.value)
    })
    for (const role of ['pitch', 'speed', 'noise', 'bass', 'presence', 'dynamics']) {
      this.$(role).addEventListener('input', () => {
        this.$('preset').value = 'custom'
        this.renderValues()
        this.markPreviewStale()
      })
    }
    this.$('normalize').addEventListener('change', () => {
      this.$('preset').value = 'custom'
      this.markPreviewStale()
    })
    this.$('save-source').addEventListener('click', () => this.downloadSource())
    this.$('save-processed').addEventListener('click', () => this.downloadProcessed())
    this.$('create').addEventListener('click', () => this.create())
  }

  methodLabel(method = this.$('method').value) {
    return method === 'signalsmith' ? this.labels.signalsmith : this.labels.ffmpeg
  }

  options() {
    return {
      method: this.$('method').value,
      preset: this.$('preset').value,
      pitch_semitones: Number(this.$('pitch').value),
      speed_factor: Number(this.$('speed').value),
      noise_reduction_db: Number(this.$('noise').value),
      bass_db: Number(this.$('bass').value),
      presence_db: Number(this.$('presence').value),
      dynamics: Number(this.$('dynamics').value),
      normalize_loudness: this.$('normalize').checked,
    }
  }

  renderValues() {
    const pitch = Number(this.$('pitch').value)
    this.$('pitch-value').textContent = `${pitch > 0 ? '+' : ''}${pitch.toFixed(1).replace(/\.0$/, '')} st`
    this.$('speed-value').textContent = `${Number(this.$('speed').value).toFixed(2)}x`
    this.$('noise-value').textContent = `${Number(this.$('noise').value).toFixed(1).replace(/\.0$/, '')} dB`
    this.$('bass-value').textContent = signedDecibels(this.$('bass').value)
    this.$('presence-value').textContent = signedDecibels(this.$('presence').value)
    this.$('dynamics-value').textContent = `${this.$('dynamics').value}%`
  }

  renderMethod() {
    this.$('signalsmith-controls').hidden = this.$('method').value !== 'signalsmith'
    if (!this.output.currentFile()) this.$('method-badge').textContent = this.methodLabel()
  }

  applyPreset(name, { invalidate = true } = {}) {
    const preset = PRESETS[name]
    if (!preset) return
    for (const [key, role] of Object.entries({
      pitch_semitones: 'pitch',
      speed_factor: 'speed',
      noise_reduction_db: 'noise',
      bass_db: 'bass',
      presence_db: 'presence',
      dynamics: 'dynamics',
    })) this.$(role).value = preset[key]
    this.$('normalize').checked = preset.normalize_loudness
    this.renderValues()
    if (invalidate) this.markPreviewStale()
  }

  applyOptions(options = {}, { markStale = true } = {}) {
    if (options.method) this.$('method').value = options.method
    if (options.preset) this.$('preset').value = options.preset
    for (const [key, role] of Object.entries({
      pitch_semitones: 'pitch',
      speed_factor: 'speed',
      noise_reduction_db: 'noise',
      bass_db: 'bass',
      presence_db: 'presence',
      dynamics: 'dynamics',
    })) {
      if (options[key] !== undefined) this.$(role).value = options[key]
    }
    if (options.normalize_loudness !== undefined) this.$('normalize').checked = Boolean(options.normalize_loudness)
    this.renderValues()
    this.renderMethod()
    if (markStale) this.markPreviewStale()
  }

  setSource(blob, extension) {
    this.source = { blob, extension }
    this.resetPreview()
    this.root.hidden = false
  }

  clear({ hide = true } = {}) {
    this.source = null
    this.resetPreview()
    if (this.adjustments) this.adjustments.open = false
    if (hide) this.root.hidden = true
  }

  resetPreview() {
    this.output.clear()
    this.processedOptions = null
    this.$('preview').hidden = true
    this.$('status').hidden = true
    this.renderStale(false)
  }

  renderStale(stale) {
    this.$('preview').classList.toggle('is-stale', Boolean(stale))
    this.$('stale-badge').hidden = !stale
  }

  markPreviewStale() {
    if (!this.output.currentFile()) return
    this.renderStale(true)
  }

  downloadSource() {
    if (!this.source) return this.onError(new Error(this.labels.generateFirst))
    const extension = String(this.source.extension || 'wav').replace(/[^a-z0-9]/gi, '') || 'wav'
    const url = URL.createObjectURL(this.source.blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `voxcpmtts-original.${extension}`
    anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  downloadProcessed() {
    if (!this.output.currentFile()) return this.onError(new Error(this.labels.processFirst))
    this.output.download()
  }

  snapshot() {
    const processed = this.output.currentFile()
    return {
      source: this.source ? { ...this.source } : null,
      processed: processed ? { blob: processed, extension: processed.name.split('.').pop() || 'wav' } : null,
      options: this.options(),
      processedOptions: this.processedOptions,
      stale: this.$('preview').classList.contains('is-stale'),
    }
  }

  async restore({ source = null, processed = null, options = {}, processedOptions = null, stale = false } = {}) {
    this.clear()
    this.applyOptions(options, { markStale: false })
    if (!source?.blob) return
    this.setSource(source.blob, source.extension)
    if (!processed?.blob) return
    await this.output.load(processed.blob, `voxcpmtts-processed.${processed.extension || 'wav'}`)
    this.processedOptions = processedOptions || options
    this.$('method-badge').textContent = this.methodLabel(this.processedOptions.method)
    this.$('preview').hidden = false
    this.renderStale(stale)
  }

  async create() {
    if (!this.source) return this.onError(new Error(this.labels.generateFirst))
    const button = this.$('create')
    const status = this.$('status')
    const options = this.options()
    button.disabled = true
    status.hidden = false
    status.dataset.tone = 'neutral'
    status.textContent = this.labels.processing.replace('{method}', this.methodLabel(options.method))
    try {
      const form = new FormData()
      form.append('audio', this.source.blob, `voxcpmtts-source.${this.source.extension || 'wav'}`)
      form.append('options', JSON.stringify(options))
      const response = await fetch('/tts/postprocess-upload', { method: 'POST', body: form })
      if (!response.ok) throw new Error(await this.responseError(response))
      const blob = await response.blob()
      await this.output.load(blob, 'voxcpmtts-processed.wav')
      this.processedOptions = options
      this.$('method-badge').textContent = this.methodLabel(options.method)
      this.$('preview').hidden = false
      this.renderStale(false)
      status.dataset.tone = 'success'
      status.textContent = this.labels.complete
    } catch (error) {
      status.hidden = false
      status.dataset.tone = 'error'
      status.textContent = error instanceof Error ? error.message : String(error)
      this.markPreviewStale()
      this.onError(error)
    } finally {
      button.disabled = false
    }
  }

  destroy() {
    document.removeEventListener('pointerdown', this.handleDocumentPointerDown)
    document.removeEventListener('keydown', this.handleDocumentKeyDown)
    this.output.destroy()
  }
}
