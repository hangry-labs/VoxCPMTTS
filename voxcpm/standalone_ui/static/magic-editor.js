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
  constructor(root, { onChange } = {}) {
    this.root = root
    this.canvas = root.querySelector('#magic-editor')
    this.voiceControl = root.querySelector('#magic-voice-control')
    this.directionControl = root.querySelector('#magic-direction-control')
    this.customDirection = root.querySelector('#magic-custom-direction')
    this.customDirectionField = root.querySelector('.magic-custom-direction-field')
    this.summary = root.querySelector('#magic-summary')
    this.onChange = onChange || (() => {})
    this.nextId = 1
    this.voices = []
    this.maxBreakMs = 10_000
    this.supportsVoice = true
    this.supportsDirection = true
    this.blocks = [this.createSpeech(t('composer.defaultText', {}, 'VoxCPM2 generates natural multilingual speech with voice design and cloning.'))]
    this.activeId = this.blocks[0].id
    this.bind()
    this.render()
  }

  createSpeech(text = '') {
    return { id: this.nextId++, type: 'speech', text, voice: '', direction: '' }
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
      block.voice = this.voiceControl.value
      if (!block.voice) block.direction = ''
      this.render()
      this.changed()
    })
    this.directionControl.addEventListener('change', () => {
      const block = this.activeSpeech()
      if (!block) return
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
      block.direction = this.customDirection.value.trim()
      this.renderBlocks()
      this.changed()
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
      this.applyBlockAction(Number(blockElement.dataset.magicId), action)
    })
    this.canvas.addEventListener('input', (event) => {
      const blockElement = event.target.closest('[data-magic-id]')
      const block = this.block(Number(blockElement?.dataset.magicId))
      if (!block) return
      if (event.target.matches('textarea')) block.text = event.target.value
      if (event.target.matches('input[type="number"]')) {
        block.milliseconds = Math.min(this.maxBreakMs, Math.max(0, Number(event.target.value) || 0))
      }
      this.updateSummary()
      this.changed()
    })
  }

  setVoices(profiles) {
    this.voices = profiles.map((profile) => ({ id: profile.id, label: profile.id }))
    const known = new Set(this.voices.map((voice) => voice.id))
    for (const block of this.blocks) {
      if (block.type === 'speech' && block.voice && !known.has(block.voice)) {
        this.voices.push({ id: block.voice, label: block.voice })
        known.add(block.voice)
      }
    }
    this.renderVoiceOptions()
    this.syncToolbar()
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

  applyBlockAction(id, action) {
    const index = this.blocks.findIndex((item) => item.id === id)
    if (index < 0) return
    if (action === 'up' && index > 0) [this.blocks[index - 1], this.blocks[index]] = [this.blocks[index], this.blocks[index - 1]]
    if (action === 'down' && index < this.blocks.length - 1) [this.blocks[index + 1], this.blocks[index]] = [this.blocks[index], this.blocks[index + 1]]
    if (action === 'remove') {
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
    actions.append(
      button('icon-chevron-up', t('magic.moveUp', {}, 'Move turn up'), 'up'),
      button('icon-chevron-down', t('magic.moveDown', {}, 'Move turn down'), 'down'),
      button('icon-x', t('magic.remove', {}, 'Remove block'), 'remove'),
    )
    actions.children[0].disabled = index === 0
    actions.children[1].disabled = index === this.blocks.length - 1
    heading.append(metadata, actions)

    const textarea = document.createElement('textarea')
    textarea.rows = 2
    textarea.value = block.text
    textarea.placeholder = t('magic.turnPlaceholder', {}, 'Write this turn...')
    textarea.setAttribute('aria-label', t('magic.turnLabel', { number: index + 1 }, `Speech turn ${index + 1}`))
    article.append(heading, textarea)
    return article
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

  toSSMLH() {
    return serializeMagicDocument(this.blocks)
  }
}
