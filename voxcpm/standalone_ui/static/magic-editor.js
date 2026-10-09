import { t } from './i18n.js'
import { serializeMagicDocument } from './magic-document.js'

const DEFAULT_BREAK_MS = 300

const DIRECTIONS = [
  ['', 'magic.directionNone', 'Natural'],
  ['Calm and reassuring', 'magic.directionCalm', 'Calm'],
  ['Energetic and upbeat', 'magic.directionEnergetic', 'Energetic'],
  ['Warm and confident', 'magic.directionWarm', 'Warm'],
  ['Authoritative and precise', 'magic.directionAuthoritative', 'Authoritative'],
  ['Urgent and alarmed', 'magic.directionUrgent', 'Urgent'],
  ['__custom__', 'magic.directionCustom', 'Custom...'],
]

const EXPRESSIVE_CUES = [
  ['', 'magic.cuePlaceholder', 'Add cue...'],
  ['[sigh]', 'magic.cueSigh', 'Sigh'],
  ['[laughing]', 'magic.cueLaughing', 'Laughing'],
  ['[Uhm]', 'magic.cueThinking', 'Thinking'],
  ['[Shh]', 'magic.cueHush', 'Hush'],
  ['[Question-en]', 'magic.cueQuestion', 'Question'],
  ['[Surprise-wa]', 'magic.cueSurprise', 'Surprise'],
  ['[Dissatisfaction-hnn]', 'magic.cueDissatisfaction', 'Dissatisfaction'],
]

function button(icon, title, action) {
  const control = document.createElement('button')
  control.type = 'button'
  control.className = 'icon-button magic-block-action'
  control.dataset.magicAction = action
  control.title = title
  control.setAttribute('aria-label', title)
  const glyph = document.createElement('i')
  glyph.className = icon
  control.append(glyph)
  return control
}

export class MagicEditor {
  constructor(root, { onChange, onPreview, onSaveCharacter, onError } = {}) {
    this.root = root
    this.canvas = root.querySelector('#magic-editor')
    this.voiceControl = root.querySelector('#magic-voice-control')
    this.directionControl = root.querySelector('#magic-direction-control')
    this.customDirection = root.querySelector('#magic-custom-direction')
    this.customDirectionField = root.querySelector('.magic-custom-direction-field')
    this.expressionControl = root.querySelector('#magic-expression-control')
    this.summary = root.querySelector('#magic-summary')
    this.onChange = onChange || (() => {})
    this.onPreview = onPreview || (async () => null)
    this.onSaveCharacter = onSaveCharacter || (async () => {})
    this.onError = onError || (() => {})
    this.nextId = 1
    this.voices = []
    this.defaultVoice = null
    this.previewBusyId = null
    this.playingId = null
    this.previewAudio = new Audio()
    this.previewAudio.addEventListener('ended', () => {
      this.playingId = null
      this.renderBlocks()
    })
    this.maxBreakMs = 10_000
    this.supportsVoice = true
    this.supportsDirection = true
    this.blocks = [this.createSpeech(t('composer.defaultText', {}, 'VoxCPM2 generates natural multilingual speech with voice design and cloning.'))]
    this.activeId = this.blocks[0].id
    this.bind()
    this.render()
  }

  createSpeech(text = '') {
    return { id: this.nextId++, type: 'speech', text, voice: '', direction: '', preview: null, captureSeed: null }
  }

  createBreak(milliseconds = DEFAULT_BREAK_MS) {
    return { id: this.nextId++, type: 'break', milliseconds }
  }

  bind() {
    this.root.querySelector('#magic-add-turn').addEventListener('click', () => this.insertAfterActive(this.createSpeech()))
    this.root.querySelector('#magic-add-pause').addEventListener('click', () => this.insertAfterActive(this.createBreak()))
    this.voiceControl.addEventListener('change', () => {
      const block = this.activeSpeech()
      if (!block) return
      this.invalidateBlock(block)
      block.voice = this.voiceControl.value
      if (!block.voice) block.direction = ''
      this.render()
      this.changed()
    })
    this.directionControl.addEventListener('change', () => {
      const block = this.activeSpeech()
      if (!block) return
      this.invalidateBlock(block)
      const custom = this.directionControl.value === '__custom__'
      this.customDirectionField.hidden = !custom
      block.direction = custom ? this.customDirection.value.trim() : this.directionControl.value
      this.renderBlocks()
      this.changed()
      if (custom) this.customDirection.focus()
    })
    this.customDirection.addEventListener('input', () => {
      const block = this.activeSpeech()
      if (!block) return
      this.invalidateBlock(block)
      block.direction = this.customDirection.value.trim()
      this.renderBlocks()
      this.changed()
    })
    this.expressionControl.addEventListener('change', () => {
      const cue = this.expressionControl.value
      this.expressionControl.value = ''
      if (cue) this.insertCue(cue)
    })
    this.canvas.addEventListener('focusin', (event) => {
      const block = event.target.closest('[data-magic-id]')
      if (block) this.activate(Number(block.dataset.magicId))
    })
    this.canvas.addEventListener('click', (event) => {
      const blockElement = event.target.closest('[data-magic-id]')
      if (blockElement) this.activate(Number(blockElement.dataset.magicId))
      const action = event.target.closest('[data-magic-action]')?.dataset.magicAction
      if (!action || !blockElement) return
      this.applyBlockAction(Number(blockElement.dataset.magicId), action).catch(this.onError)
    })
    this.canvas.addEventListener('input', (event) => {
      const blockElement = event.target.closest('[data-magic-id]')
      const block = this.block(Number(blockElement?.dataset.magicId))
      if (!block) return
      if (event.target.matches('textarea')) {
        const generatedStateChanged = Boolean(block.preview || Number.isInteger(block.captureSeed))
        block.text = event.target.value
        this.invalidateBlock(block)
        if (generatedStateChanged) this.renderBlocks()
      }
      if (event.target.matches('input[type="number"]')) {
        block.milliseconds = Math.min(this.maxBreakMs, Math.max(0, Number(event.target.value) || 0))
      }
      this.updateSummary()
      this.changed()
    })
  }

  setVoices(profiles) {
    this.voices = profiles.map((profile) => ({
      id: profile.id,
      label: profile.id,
      portraitUrl: profile.portrait_url || '',
      version: profile.created_at || 'current',
    }))
    const known = new Set(this.voices.map((voice) => voice.id))
    for (const block of this.blocks) {
      if (block.type === 'speech' && block.voice && !known.has(block.voice)) {
        this.voices.push({ id: block.voice, label: block.voice, portraitUrl: '', version: 'current' })
        known.add(block.voice)
      }
    }
    this.renderVoiceOptions()
    this.renderBlocks()
    this.syncToolbar()
  }

  setDefaultVoice(profile) {
    const next = profile ? {
      id: profile.id,
      label: profile.id,
      portraitUrl: profile.portrait_url || '',
      version: profile.created_at || 'current',
    } : null
    if (next?.id === this.defaultVoice?.id
      && next?.portraitUrl === this.defaultVoice?.portraitUrl
      && next?.version === this.defaultVoice?.version) return
    this.defaultVoice = next
    this.blocks.filter((block) => block.type === 'speech' && !block.voice).forEach((block) => this.invalidateBlock(block))
    this.renderBlocks()
  }

  setCapabilities(capabilities) {
    const elements = new Set(capabilities?.ssml?.elements || [])
    this.supportsVoice = elements.has('voice')
    this.supportsDirection = Boolean(capabilities?.ssml_h?.turn_direction?.supported)
    this.root.querySelector('#magic-add-pause').disabled = !elements.has('break')
    this.maxBreakMs = Number(capabilities?.limits?.break_ms) || this.maxBreakMs
    this.renderBlocks()
    this.syncToolbar()
  }

  loadSample(text) {
    this.blocks.forEach((block) => this.clearPreview(block))
    const first = this.voices[0]?.id || ''
    const second = this.voices[1]?.id || first
    if (first) {
      this.blocks = [
        { ...this.createSpeech(text), voice: first, direction: 'Warm and confident' },
        this.createBreak(),
        { ...this.createSpeech(t('magic.sampleReply', {}, 'Everything is ready. We can begin when you are.')), voice: second, direction: 'Calm and reassuring' },
      ]
    } else {
      this.blocks = [this.createSpeech(text)]
    }
    this.activeId = this.blocks[0].id
    this.render()
    this.changed()
  }

  insertAfterActive(block) {
    const index = Math.max(0, this.blocks.findIndex((item) => item.id === this.activeId))
    this.blocks.splice(index + 1, 0, block)
    this.activeId = block.id
    this.render()
    this.changed()
    this.canvas.querySelector(`[data-magic-id="${block.id}"] textarea`)?.focus()
  }

  async applyBlockAction(id, action) {
    const index = this.blocks.findIndex((item) => item.id === id)
    if (index < 0) return
    if (action === 'preview') {
      await this.togglePreview(id)
      return
    }
    if (action === 'save-character') {
      await this.onSaveCharacter({ ...this.blocks[index] })
      return
    }
    if (action === 'up' && index > 0) [this.blocks[index - 1], this.blocks[index]] = [this.blocks[index], this.blocks[index - 1]]
    if (action === 'down' && index < this.blocks.length - 1) [this.blocks[index + 1], this.blocks[index]] = [this.blocks[index], this.blocks[index + 1]]
    if (action === 'remove') {
      this.clearPreview(this.blocks[index])
      if (this.blocks.length === 1) {
        this.blocks = [this.createSpeech()]
      } else {
        this.blocks.splice(index, 1)
      }
      this.activeId = this.blocks[Math.min(index, this.blocks.length - 1)].id
    }
    this.render()
    this.changed()
  }

  insertCue(cue) {
    const block = this.activeSpeech()
    if (!block) return
    const textarea = this.canvas.querySelector(`[data-magic-id="${block.id}"] textarea`)
    const start = Number.isInteger(textarea?.selectionStart) ? textarea.selectionStart : block.text.length
    const end = Number.isInteger(textarea?.selectionEnd) ? textarea.selectionEnd : start
    const before = block.text.slice(0, start)
    const after = block.text.slice(end)
    const prefix = before && !/\s$/.test(before) ? ' ' : ''
    const suffix = after && !/^\s/.test(after) ? ' ' : ''
    block.text = `${before}${prefix}${cue}${suffix}${after}`
    this.invalidateBlock(block)
    this.renderBlocks()
    const updated = this.canvas.querySelector(`[data-magic-id="${block.id}"] textarea`)
    const cursor = before.length + prefix.length + cue.length + suffix.length
    updated?.focus()
    updated?.setSelectionRange(cursor, cursor)
    this.updateSummary()
    this.changed()
  }

  clearPreview(block) {
    if (!block?.preview) return
    if (this.playingId === block.id) {
      this.previewAudio.pause()
      this.playingId = null
    }
    URL.revokeObjectURL(block.preview.url)
    block.preview = null
  }

  invalidateBlock(block) {
    this.clearPreview(block)
    block.captureSeed = null
  }

  async ensurePreview(id, { play = false } = {}) {
    const block = this.block(id)
    if (!block || block.type !== 'speech' || !block.text.trim()) return null
    if (!block.preview) {
      if (this.previewBusyId !== null) return null
      this.previewBusyId = id
      this.renderBlocks()
      try {
        const result = await this.onPreview({ ...block })
        if (!result?.blob) return null
        this.clearPreview(block)
        block.preview = { ...result, url: URL.createObjectURL(result.blob) }
      } finally {
        this.previewBusyId = null
        this.renderBlocks()
      }
    }
    if (play && block.preview) await this.playPreview(block)
    return block.preview
  }

  async togglePreview(id) {
    if (this.playingId === id) {
      this.previewAudio.pause()
      this.playingId = null
      this.renderBlocks()
      return
    }
    const preview = await this.ensurePreview(id)
    const block = this.block(id)
    if (preview && block) await this.playPreview(block)
  }

  async playPreview(block) {
    this.previewAudio.pause()
    this.previewAudio.src = block.preview.url
    this.playingId = block.id
    this.renderBlocks()
    try {
      await this.previewAudio.play()
    } catch (error) {
      this.playingId = null
      this.renderBlocks()
      if (error?.name !== 'AbortError') this.onError(error)
    }
  }

  assignVoice(id, voice) {
    const block = this.block(id)
    if (!block || block.type !== 'speech') return
    block.voice = voice
    block.direction = ''
    block.captureSeed = null
    this.activeId = id
    this.render()
    this.changed()
  }

  markFullGeneration(seed) {
    if (!Number.isInteger(seed)) return
    let speechIndex = 0
    for (const block of this.blocks) {
      if (block.type === 'speech') this.clearPreview(block)
      if (block.type !== 'speech' || !block.text.trim()) {
        if (block.type === 'speech') block.captureSeed = null
        continue
      }
      block.captureSeed = (seed + speechIndex) % (2 ** 32)
      speechIndex += 1
    }
    this.renderBlocks()
  }

  activate(id) {
    if (this.activeId === id) return
    this.activeId = id
    this.canvas.querySelectorAll('[data-magic-id]').forEach((element) => {
      element.classList.toggle('active', Number(element.dataset.magicId) === id)
    })
    this.syncToolbar()
  }

  block(id) {
    return this.blocks.find((item) => item.id === id)
  }

  activeSpeech() {
    const block = this.block(this.activeId)
    return block?.type === 'speech' ? block : null
  }

  render() {
    this.renderVoiceOptions()
    this.renderDirectionOptions()
    this.renderExpressionOptions()
    this.renderBlocks()
    this.syncToolbar()
    this.updateSummary()
  }

  renderVoiceOptions() {
    const selected = this.voiceControl.value
    const options = [{ id: '', label: t('magic.defaultVoice', {}, 'Default voice') }, ...this.voices]
    this.voiceControl.replaceChildren(...options.map((voice) => {
      const option = document.createElement('option')
      option.value = voice.id
      option.textContent = voice.label
      return option
    }))
    if (options.some((voice) => voice.id === selected)) this.voiceControl.value = selected
  }

  renderDirectionOptions() {
    if (this.directionControl.options.length) return
    this.directionControl.replaceChildren(...DIRECTIONS.map(([value, key, fallback]) => {
      const option = document.createElement('option')
      option.value = value
      option.textContent = t(key, {}, fallback)
      return option
    }))
  }

  renderExpressionOptions() {
    if (this.expressionControl.options.length) return
    this.expressionControl.replaceChildren(...EXPRESSIVE_CUES.map(([value, key, fallback]) => {
      const option = document.createElement('option')
      option.value = value
      option.textContent = t(key, {}, fallback)
      return option
    }))
  }

  renderBlocks() {
    const activeElement = document.activeElement
    const activeId = Number(activeElement?.closest?.('[data-magic-id]')?.dataset.magicId)
    const selectionStart = activeElement?.selectionStart
    const selectionEnd = activeElement?.selectionEnd
    this.canvas.replaceChildren(...this.blocks.map((block, index) => (
      block.type === 'break' ? this.renderBreak(block, index) : this.renderSpeech(block, index)
    )))
    if (activeId) {
      const restored = this.canvas.querySelector(`[data-magic-id="${activeId}"] textarea`)
      if (restored) {
        restored.focus()
        if (Number.isInteger(selectionStart)) restored.setSelectionRange(selectionStart, selectionEnd)
      }
    }
  }

  renderSpeech(block, index) {
    const article = document.createElement('article')
    article.className = `magic-turn${block.id === this.activeId ? ' active' : ''}`
    article.dataset.magicId = String(block.id)

    const heading = document.createElement('div')
    heading.className = 'magic-block-heading'
    const metadata = document.createElement('div')
    metadata.className = 'magic-turn-metadata'
    const voice = document.createElement('span')
    voice.className = 'magic-chip voice'
    voice.textContent = block.voice || t('magic.defaultVoice', {}, 'Default voice')
    metadata.append(voice)
    if (block.direction) {
      const direction = document.createElement('span')
      direction.className = 'magic-chip direction'
      direction.textContent = block.direction
      metadata.append(direction)
    }
    const actions = document.createElement('div')
    actions.className = 'magic-block-actions'
    const previewing = this.previewBusyId === block.id
    const playing = this.playingId === block.id
    const preview = button(
      previewing ? 'icon-refresh-cw' : playing ? 'icon-pause' : 'icon-play',
      previewing
        ? t('magic.previewGenerating', {}, 'Generating turn preview')
        : playing
          ? t('magic.previewPause', {}, 'Pause turn preview')
          : block.preview
            ? t('magic.previewPlay', {}, 'Play turn preview')
            : t('magic.previewGenerate', {}, 'Generate this turn'),
      'preview',
    )
    preview.classList.add('magic-preview-action')
    preview.classList.toggle('ready', Boolean(block.preview))
    preview.classList.toggle('loading', previewing)
    preview.disabled = !block.text.trim() || (this.previewBusyId !== null && !previewing)
    actions.append(
      preview,
      button('icon-chevron-up', t('magic.moveUp', {}, 'Move turn up'), 'up'),
      button('icon-chevron-down', t('magic.moveDown', {}, 'Move turn down'), 'down'),
      button('icon-x', t('magic.remove', {}, 'Remove block'), 'remove'),
    )
    actions.children[1].disabled = index === 0
    actions.children[2].disabled = index === this.blocks.length - 1
    heading.append(metadata, actions)

    const textarea = document.createElement('textarea')
    textarea.rows = 2
    textarea.value = block.text
    textarea.placeholder = t('magic.turnPlaceholder', {}, 'Write this turn...')
    textarea.setAttribute('aria-label', t('magic.turnLabel', { number: index + 1 }, `Speech turn ${index + 1}`))
    const content = document.createElement('div')
    content.className = 'magic-turn-content'
    content.append(this.renderPortrait(block), textarea)
    article.append(heading, content)
    return article
  }

  renderPortrait(block) {
    const voice = block.voice
      ? this.voices.find((item) => item.id === block.voice)
      : this.defaultVoice
    const portrait = document.createElement(voice ? 'div' : 'button')
    portrait.className = 'magic-turn-portrait'
    if (voice) {
      portrait.title = voice.label
      if (voice.portraitUrl) {
        const image = document.createElement('img')
        image.src = `${voice.portraitUrl}?v=${encodeURIComponent(voice.version)}`
        image.alt = t('profiles.portraitAlt', { name: voice.label }, `${voice.label} portrait`)
        image.loading = 'lazy'
        portrait.append(image)
      } else {
        const placeholder = document.createElement('span')
        placeholder.textContent = '?'
        placeholder.setAttribute('aria-hidden', 'true')
        portrait.append(placeholder)
      }
      return portrait
    }

    portrait.type = 'button'
    portrait.dataset.magicAction = 'save-character'
    const canSave = Boolean(block.preview || Number.isInteger(block.captureSeed))
    portrait.disabled = !canSave || this.previewBusyId !== null
    portrait.classList.toggle('ready', canSave)
    portrait.title = canSave
      ? t('magic.saveCharacter', {}, 'Save this generated voice as a character')
      : t('magic.previewToSave', {}, 'Generate this turn before saving its voice')
    portrait.setAttribute('aria-label', portrait.title)
    const plus = document.createElement('span')
    plus.textContent = '+'
    plus.setAttribute('aria-hidden', 'true')
    portrait.append(plus)
    return portrait
  }

  renderBreak(block, index) {
    const element = document.createElement('div')
    element.className = `magic-pause${block.id === this.activeId ? ' active' : ''}`
    element.dataset.magicId = String(block.id)
    const line = document.createElement('span')
    line.className = 'magic-pause-line'
    const label = document.createElement('label')
    label.append(document.createTextNode(t('magic.pause', {}, 'Pause')))
    const input = document.createElement('input')
    input.type = 'number'
    input.min = '0'
    input.max = String(this.maxBreakMs)
    input.step = '50'
    input.value = String(block.milliseconds)
    input.setAttribute('aria-label', t('magic.pauseDuration', {}, 'Pause duration in milliseconds'))
    const unit = document.createElement('span')
    unit.textContent = 'ms'
    const actions = document.createElement('div')
    actions.className = 'magic-block-actions'
    actions.append(
      button('icon-chevron-up', t('magic.moveUp', {}, 'Move turn up'), 'up'),
      button('icon-chevron-down', t('magic.moveDown', {}, 'Move turn down'), 'down'),
      button('icon-x', t('magic.remove', {}, 'Remove block'), 'remove'),
    )
    actions.children[0].disabled = index === 0
    actions.children[1].disabled = index === this.blocks.length - 1
    element.append(line, label, input, unit, actions)
    return element
  }

  syncToolbar() {
    const block = this.activeSpeech()
    this.voiceControl.disabled = !block || !this.supportsVoice
    this.directionControl.disabled = !block || !block.voice || !this.supportsVoice || !this.supportsDirection
    this.expressionControl.disabled = !block
    if (!block) {
      this.customDirectionField.hidden = true
      return
    }
    this.voiceControl.value = block.voice
    const predefined = DIRECTIONS.some(([value]) => value === block.direction)
    this.directionControl.value = predefined ? block.direction : '__custom__'
    this.customDirection.value = predefined ? '' : block.direction
    this.customDirectionField.hidden = predefined
  }

  plainText() {
    return this.blocks
      .filter((block) => block.type === 'speech')
      .map((block) => block.text.trim())
      .filter(Boolean)
      .join('\n')
  }

  stats() {
    const text = this.plainText()
    return {
      characters: text.length,
      words: text ? text.split(/\s+/).length : 0,
      turns: this.blocks.filter((block) => block.type === 'speech' && block.text.trim()).length,
    }
  }

  updateSummary() {
    const stats = this.stats()
    this.summary.textContent = t('magic.summary', { turns: stats.turns }, `${stats.turns} turns`)
  }

  changed() {
    this.onChange(this)
  }

  toSSMLH(blocks = this.blocks) {
    return serializeMagicDocument(blocks)
  }
}
