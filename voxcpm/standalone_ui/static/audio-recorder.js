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
  constructor({ button, status, onFile, onError, labels }) {
    this.button = button
    this.status = status
    this.onFile = onFile
    this.onError = onError
    this.labels = labels
    this.stream = null
    this.recorder = null
    this.chunks = []
    this.startedAt = 0
    this.timer = null

    this.button.addEventListener('click', () => this.toggle())
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !window.AudioContext) {
      this.button.disabled = true
      this.status.textContent = this.labels.unavailable
      this.button.title = this.labels.unavailable
    }
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
    this.stream = null
    this.recorder = null
    this.chunks = []
  }

  stop() {
    if (this.recorder?.state === 'recording') this.recorder.stop()
  }
}
