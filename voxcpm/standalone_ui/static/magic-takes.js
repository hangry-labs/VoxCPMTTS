const BUNDLE_MEDIA_TYPE = 'application/vnd.hangrylabs.magic-takes'

function speechBlocks(blocks) {
  return blocks.filter((block) => block.type === 'speech' && block.text.trim())
}

function takeFile(block) {
  return new File(
    [block.preview.blob],
    `voxcpmtts-turn-${block.id}.wav`,
    { type: block.preview.blob.type || 'audio/wav' },
  )
}

async function unpackBundle(response) {
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength < 4) throw new Error('Magic take response is incomplete.')
  const view = new DataView(buffer)
  const manifestLength = view.getUint32(0, true)
  const manifestEnd = 4 + manifestLength
  if (manifestEnd > buffer.byteLength) throw new Error('Magic take manifest is incomplete.')
  const manifest = JSON.parse(new TextDecoder().decode(buffer.slice(4, manifestEnd)))
  if (manifest.version !== 1 || !Array.isArray(manifest.takes) || !manifest.output) {
    throw new Error('Magic take response uses an unsupported format.')
  }
  const payload = buffer.slice(manifestEnd)
  const extract = (item) => {
    const start = Number(item.offset)
    const end = start + Number(item.length)
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > payload.byteLength) {
      throw new Error('Magic take response contains an invalid audio range.')
    }
    return new Blob([payload.slice(start, end)], { type: item.media_type || 'application/octet-stream' })
  }
  return {
    requestSeed: Number(manifest.request_seed),
    sampleRate: Number(manifest.sample_rate),
    takes: new Map(manifest.takes.map((item) => [String(item.id), {
      blob: extract(item),
      extension: item.extension || 'wav',
      seed: Number(item.seed),
    }])),
    output: {
      blob: extract(manifest.output),
      extension: manifest.output.extension || 'wav',
    },
  }
}

export class MagicTakeStudio {
  constructor({ responseError }) {
    this.responseError = responseError
  }

  async render(payload, blocks, { signal } = {}) {
    const speech = speechBlocks(blocks)
    if (!speech.length) throw new Error('Add at least one speech turn before generating.')
    const retained = speech.filter((block) => block.locked && block.preview)
    const form = new FormData()
    form.append('payload', JSON.stringify(payload))
    form.append('speech_ids', JSON.stringify(speech.map((block) => String(block.id))))
    form.append('retained', JSON.stringify(retained.map((block) => ({
      id: String(block.id),
      seed: Number(block.captureSeed) || 0,
    }))))
    retained.forEach((block) => form.append('takes', takeFile(block)))
    const response = await fetch('/tts/magic/render', { method: 'POST', body: form, signal })
    if (!response.ok) throw new Error(await this.responseError(response))
    if (!response.headers.get('Content-Type')?.startsWith(BUNDLE_MEDIA_TYPE)) {
      throw new Error('Magic render returned an unexpected response type.')
    }
    return unpackBundle(response)
  }

  async take(payload, speechIndex, { signal } = {}) {
    const response = await fetch('/tts/magic/take', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ payload, speech_index: speechIndex }),
      signal,
    })
    if (!response.ok) throw new Error(await this.responseError(response))
    return {
      blob: await response.blob(),
      extension: response.headers.get('X-VoxCPM-Format') || 'wav',
      seed: Number(response.headers.get('X-VoxCPM-Seed')),
    }
  }

  async assemble(blocks, { outputFormat = 'wav', normalizeLoudness = false, signal } = {}) {
    const speech = speechBlocks(blocks)
    if (!speech.length || speech.some((block) => !block.preview)) {
      throw new Error('Generate every speech turn before assembling the dialogue.')
    }
    const timeline = blocks.flatMap((block) => {
      if (block.type === 'break') return [{ type: 'break', milliseconds: block.milliseconds }]
      return block.text.trim() ? [{ type: 'speech', id: String(block.id) }] : []
    })
    const form = new FormData()
    form.append('options', JSON.stringify({
      items: timeline,
      output_format: outputFormat,
      normalize_loudness: normalizeLoudness,
    }))
    form.append('take_metadata', JSON.stringify(speech.map((block) => ({
      id: String(block.id),
      seed: Number(block.captureSeed) || 0,
    }))))
    speech.forEach((block) => form.append('takes', takeFile(block)))
    const response = await fetch('/tts/magic/assemble-upload', { method: 'POST', body: form, signal })
    if (!response.ok) throw new Error(await this.responseError(response))
    return {
      blob: await response.blob(),
      extension: response.headers.get('X-VoxCPM-Format') || outputFormat,
    }
  }
}
