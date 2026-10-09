import { t } from './i18n.js'

function formatClock(seconds) {
  const safe = Math.max(0, Number(seconds) || 0)
  return `${Math.floor(safe / 60)}:${Math.floor(safe % 60).toString().padStart(2, '0')}`
}

export class StreamWaveform {
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
    this.stateElement.textContent = t('stream.waitingFirst', {}, 'Waiting for first audio chunk')
    this.detailElement.textContent = t('stream.buffered', { size: 0 }, '0 KiB buffered')
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
    this.stateElement.textContent = totalBytes
      ? t('stream.playing', {}, 'Playing generated speech')
      : t('stream.waitingFirst', {}, 'Waiting for first audio chunk')
    this.detailElement.textContent = t(
      'stream.bufferedTime',
      { size: (totalBytes / 1024).toFixed(0), time: formatClock(currentTime) },
      `${(totalBytes / 1024).toFixed(0)} KiB buffered · ${formatClock(currentTime)}`,
    )
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
    this.stateElement.textContent = t('status.streamComplete', {}, 'Stream complete')
    this.detailElement.textContent = t('stream.received', { size: (this.totalBytes / 1024).toFixed(0) }, `${(this.totalBytes / 1024).toFixed(0)} KiB received`)
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

export class IncrementalAudioPlayback {
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
      const failed = () => { this.sourceBuffer.removeEventListener('updateend', done); reject(new Error(t('errors.streamBuffer', {}, 'Browser could not buffer streamed MP3 audio.'))) }
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
