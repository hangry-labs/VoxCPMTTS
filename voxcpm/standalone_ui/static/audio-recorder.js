import { encodeWav, fileFromBlob, formatTime } from './audio-utils.js'

function recordingName() {
  return `voice-reference-${new Date().toISOString().replace(/[:.]/g, '-')}.wav`
}

async function convertToWav(blob) {
  const context = new AudioContext()
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer())
    const channels = []
    for (let index = 0; index < decoded.numberOfChannels; index += 1) {
      channels.push(decoded.getChannelData(index))
    }
    return fileFromBlob(encodeWav(channels, decoded.sampleRate), recordingName())
  } finally {
    await context.close()
  }
}

export class AudioRecorder {
  constructor({ button, status, canvas = null, onFile, onError, labels }) {
    this.button = button
    this.status = status
    this.canvas = canvas
    this.onFile = onFile
    this.onError = onError
    this.labels = labels
    this.stream = null
    this.recorder = null
    this.chunks = []
    this.startedAt = 0
    this.timer = null
    this.animationFrame = null

    this.button.addEventListener('click', () => this.toggle())
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !window.AudioContext) {
      this.button.disabled = true
      this.status.textContent = this.labels.unavailable
      this.button.title = this.labels.unavailable
    }
    this.clearCanvas()
  }

  setButton(recording, busy = false) {
    this.button.dataset.recording = String(recording)
    this.button.setAttribute('aria-pressed', String(recording))
    this.button.disabled = busy
    this.button.innerHTML = recording
      ? `<i class="icon-square"></i><span>${this.labels.stop}</span>`
      : `<i class="icon-mic"></i><span>${this.labels.record}</span>`
  }

  updateElapsed() {
    const elapsed = Math.max(0, (performance.now() - this.startedAt) / 1000)
    this.status.textContent = this.labels.recording.replace('{time}', formatTime(elapsed))
  }

  async toggle() {
    if (this.recorder?.state === 'recording') {
      this.setButton(true, true)
      this.recorder.stop()
      return
    }
    await this.start()
  }

  async start() {
    this.setButton(false, true)
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      })
      if (this.canvas) {
        this.audioContext = new AudioContext()
        this.source = this.audioContext.createMediaStreamSource(this.stream)
        this.analyser = this.audioContext.createAnalyser()
        this.analyser.fftSize = 2048
        this.source.connect(this.analyser)
        await this.audioContext.resume()
        this.draw()
      }
      this.chunks = []
      this.recorder = new MediaRecorder(this.stream)
      this.recorder.addEventListener('dataavailable', (event) => {
        if (event.data.size) this.chunks.push(event.data)
      })
      this.recorder.addEventListener('stop', () => this.finish())
      this.recorder.start(250)
      this.startedAt = performance.now()
      this.setButton(true)
      this.updateElapsed()
      this.timer = window.setInterval(() => this.updateElapsed(), 250)
    } catch (error) {
      this.cleanup()
      this.setButton(false)
      this.status.textContent = this.labels.ready
      this.onError(error)
    }
  }

  async finish() {
    const elapsed = Math.max(0, (performance.now() - this.startedAt) / 1000)
    window.clearInterval(this.timer)
    this.timer = null
    this.status.textContent = this.labels.processing
    try {
      const blob = new Blob(this.chunks, { type: this.recorder?.mimeType || 'audio/webm' })
      const file = await convertToWav(blob)
      await this.onFile(file)
      this.status.textContent = this.labels.readyWithTime.replace('{time}', formatTime(elapsed))
    } catch (error) {
      this.status.textContent = this.labels.ready
      this.onError(error)
    } finally {
      this.cleanup()
      this.setButton(false)
    }
  }

  cleanup() {
    window.clearInterval(this.timer)
    this.timer = null
    this.stream?.getTracks().forEach((track) => track.stop())
    cancelAnimationFrame(this.animationFrame)
    this.source?.disconnect()
    if (this.audioContext?.state !== 'closed') this.audioContext?.close().catch(() => {})
    this.audioContext = null
    this.source = null
    this.analyser = null
    this.animationFrame = null
    this.stream = null
    this.recorder = null
    this.chunks = []
    this.clearCanvas()
  }

  draw() {
    if (!this.canvas || !this.analyser) return
    const context = this.canvas.getContext('2d')
    const values = new Uint8Array(this.analyser.fftSize)
    this.analyser.getByteTimeDomainData(values)
    const ratio = window.devicePixelRatio || 1
    const width = this.canvas.width = Math.max(1, Math.round(this.canvas.clientWidth * ratio))
    const height = this.canvas.height = Math.max(1, Math.round(this.canvas.clientHeight * ratio))
    context.clearRect(0, 0, width, height)
    context.strokeStyle = '#ff7a1a'
    context.lineWidth = 2 * ratio
    context.beginPath()
    values.forEach((value, index) => {
      const x = index / (values.length - 1) * width
      const y = value / 255 * height
      if (index === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    })
    context.stroke()
    this.animationFrame = requestAnimationFrame(() => this.draw())
  }

  clearCanvas() {
    if (!this.canvas) return
    const context = this.canvas.getContext('2d')
    const ratio = window.devicePixelRatio || 1
    const width = this.canvas.width = Math.max(1, Math.round(this.canvas.clientWidth * ratio))
    const height = this.canvas.height = Math.max(1, Math.round(this.canvas.clientHeight * ratio))
    context.clearRect(0, 0, width, height)
    context.strokeStyle = '#393b43'
    context.lineWidth = ratio
    context.beginPath()
    context.moveTo(0, height / 2)
    context.lineTo(width, height / 2)
    context.stroke()
  }

  stop() {
    if (this.recorder?.state === 'recording') this.recorder.stop()
  }

  refresh() {
    if (this.recorder?.state !== 'recording') this.clearCanvas()
  }
}
