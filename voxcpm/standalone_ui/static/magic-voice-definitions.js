import { t } from './i18n.js'

const DESIGN_FIELDS = ['gender', 'age', 'pitch', 'style', 'languages', 'accent', 'dialect']

function cloneDefinitions(definitions) {
  return (definitions || []).map((definition) => ({ ...definition }))
}

export class MagicVoiceDefinitions {
  constructor(dialog, { onChange, onError } = {}) {
    this.dialog = dialog
    this.form = dialog.querySelector('#magic-definition-form')
    this.list = dialog.querySelector('#magic-definition-list')
    this.summary = dialog.querySelector('#magic-definition-summary')
    this.onChange = onChange || (() => {})
    this.onError = onError || (() => {})
    this.definitions = []
    this.selectedIndex = -1
    this.maxDefinitions = 12
    this.bind()
  }

  bind() {
    this.dialog.querySelector('#magic-definition-close').addEventListener('click', () => this.close())
    this.dialog.querySelector('#magic-definition-done').addEventListener('click', () => this.close())
    this.dialog.querySelector('#magic-definition-add').addEventListener('click', () => this.startNew())
    this.dialog.querySelector('#magic-definition-delete').addEventListener('click', () => this.removeSelected())
    this.list.addEventListener('change', () => this.select(Number(this.list.value)))
    this.form.addEventListener('submit', (event) => {
      event.preventDefault()
      try {
        this.saveForm()
      } catch (error) {
        this.onError(error)
      }
    })
    this.form.elements.scope.addEventListener('change', () => {
      const profile = this.form.elements.scope.value === 'profile'
      this.form.elements.replace.disabled = !profile
      if (!profile) this.form.elements.replace.checked = false
    })
    this.dialog.addEventListener('click', (event) => {
      if (event.target === this.dialog) this.close()
    })
    this.dialog.addEventListener('cancel', (event) => {
      event.preventDefault()
      this.close()
    })
  }

  setCapabilities(capabilities) {
    this.maxDefinitions = Number(capabilities?.limits?.voice_definitions) || 12
    this.renderList()
  }

  setDefinitions(definitions) {
    this.definitions = cloneDefinitions(definitions)
    this.selectedIndex = this.definitions.length ? 0 : -1
    this.renderList()
    this.fillForm(this.selectedIndex >= 0 ? this.definitions[this.selectedIndex] : null)
  }

  getDefinitions() {
    return cloneDefinitions(this.definitions)
  }

  show() {
    this.renderList()
    this.fillForm(this.selectedIndex >= 0 ? this.definitions[this.selectedIndex] : null)
    this.dialog.showModal()
    if (this.selectedIndex < 0) this.form.elements.name.focus()
  }

  close() {
    this.dialog.close()
  }

  startNew() {
    if (this.definitions.length >= this.maxDefinitions) {
      this.onError(new Error(t(
        'magic.definitionLimit',
        { count: this.maxDefinitions },
        `A script can contain at most ${this.maxDefinitions} character definitions.`,
      )))
      return
    }
    this.selectedIndex = -1
    this.list.value = ''
    this.fillForm(null)
    this.form.elements.name.focus()
  }

  select(index) {
    if (!Number.isInteger(index) || !this.definitions[index]) return
    this.selectedIndex = index
    this.fillForm(this.definitions[index])
    this.renderList()
  }

  fillForm(definition) {
    const value = definition || {}
    for (const element of this.form.elements) {
      if (!element.name) continue
      if (element.type === 'checkbox') element.checked = value[element.name] === true || value[element.name] === 'true'
      else element.value = value[element.name] ?? ''
    }
    if (!definition) this.form.elements.scope.value = 'request'
    this.form.elements.replace.disabled = this.form.elements.scope.value !== 'profile'
    this.dialog.querySelector('#magic-definition-delete').disabled = this.selectedIndex < 0
    this.dialog.querySelector('#magic-definition-form-title').textContent = definition
      ? t('magic.definitionEdit', { name: definition.name }, `Edit ${definition.name}`)
      : t('magic.definitionNew', {}, 'New character')
  }

  readForm() {
    const name = this.form.elements.name.value.trim()
    if (!/^[A-Za-z][A-Za-z0-9._-]{0,63}$/.test(name)) {
      throw new Error(t(
        'magic.definitionNameError',
        {},
        'Character names must start with a letter and use only letters, numbers, dots, hyphens, or underscores.',
      ))
    }
    const duplicate = this.definitions.findIndex((definition, index) => (
      index !== this.selectedIndex && definition.name.toLowerCase() === name.toLowerCase()
    ))
    if (duplicate >= 0) throw new Error(t('magic.definitionDuplicate', { name }, `Character ${name} already exists in this script.`))

    const definition = {
      name,
      scope: this.form.elements.scope.value || 'request',
      replace: this.form.elements.replace.checked,
    }
    for (const field of DESIGN_FIELDS) {
      const value = this.form.elements[field].value.trim()
      if (value) definition[field] = value
    }
    const description = this.form.elements.description.value.trim()
    const sample = this.form.elements.sample.value.trim()
    const sampleLanguage = this.form.elements.sampleLanguage.value.trim()
    const seed = this.form.elements.seed.value.trim()
    if (description) definition.description = description
    if (sample) definition.sample = sample
    if (sampleLanguage) definition.sampleLanguage = sampleLanguage
    if (seed) {
      const number = Number(seed)
      if (!Number.isInteger(number) || number < 0 || number > 4294967295) {
        throw new Error(t('magic.definitionSeedError', {}, 'Seed must be an integer between 0 and 4294967295.'))
      }
      definition.seed = String(number)
    }
    if (!description && !DESIGN_FIELDS.some((field) => definition[field])) {
      throw new Error(t('magic.definitionDesignError', {}, 'Add a description or at least one voice design property.'))
    }
    if (definition.replace && definition.scope !== 'profile') {
      throw new Error(t('magic.definitionReplaceError', {}, 'Replacement is available only for persistent profile scope.'))
    }
    return definition
  }

  saveForm() {
    const definition = this.readForm()
    if (this.selectedIndex < 0) {
      this.definitions.push(definition)
      this.selectedIndex = this.definitions.length - 1
    } else {
      this.definitions[this.selectedIndex] = definition
    }
    this.renderList()
    this.fillForm(definition)
    this.onChange(this.getDefinitions())
  }

  removeSelected() {
    if (this.selectedIndex < 0) return
    const [removed] = this.definitions.splice(this.selectedIndex, 1)
    this.selectedIndex = Math.min(this.selectedIndex, this.definitions.length - 1)
    this.renderList()
    this.fillForm(this.selectedIndex >= 0 ? this.definitions[this.selectedIndex] : null)
    this.onChange(this.getDefinitions(), { removed: removed.name })
  }

  renderList() {
    const options = this.definitions.map((definition, index) => {
      const option = document.createElement('option')
      option.value = String(index)
      option.textContent = definition.name
      return option
    })
    this.list.replaceChildren(...options)
    this.list.value = this.selectedIndex >= 0 ? String(this.selectedIndex) : ''
    this.summary.textContent = t(
      'magic.definitionSummary',
      { count: this.definitions.length, limit: this.maxDefinitions },
      `${this.definitions.length} of ${this.maxDefinitions} characters`,
    )
    this.dialog.querySelector('#magic-definition-add').disabled = this.definitions.length >= this.maxDefinitions
  }
}
