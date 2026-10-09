import { t } from './i18n.js'

const $ = (selector, root = document) => root.querySelector(selector)

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

function normalizedName(value) {
  return value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^[-_]+|[-_]+$/g, '').slice(0, 80)
}

function tags(value) {
  return [...new Set(value.split(',').map((tag) => tag.trim().toLowerCase()).filter(Boolean))]
}

function downloadDocument(documentText, filename) {
  const url = URL.createObjectURL(new Blob([documentText], { type: 'application/ssml+xml;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

export class DialogueScriptLibrary {
  constructor({ magicEditor, fetchJson, showToast, errorMessage, getInputType, setInputType, onDocumentChange }) {
    this.magicEditor = magicEditor
    this.fetchJson = fetchJson
    this.showToast = showToast
    this.errorMessage = errorMessage
    this.getInputType = getInputType
    this.setInputType = setInputType
    this.onDocumentChange = onDocumentChange
    this.scripts = []
    this.active = null
    this.dirty = false
    this.pendingDelete = null
    this.editingDetails = null
    this.suppressDirty = false
    this.bindEvents()
  }

  initialize(scripts) {
    this.scripts = scripts
    this.render()
  }

  markDirty() {
    if (this.suppressDirty || !this.active) return
    this.dirty = true
    this.render()
  }

  runWithoutDirty(callback) {
    this.suppressDirty = true
    try { return callback() } finally { this.suppressDirty = false }
  }

  renderSaveState() {
    const scriptId = normalizedName($('#script-name').value)
    const updating = Boolean(scriptId && scriptId === this.active?.id)
    $('#script-save').hidden = !scriptId
    $('span', $('#script-save')).textContent = updating
      ? t('scripts.updateShort', {}, 'Update')
      : t('scripts.saveShort', {}, 'Save')
  }

  render() {
    const query = $('#script-filter').value.trim().toLowerCase()
    const scripts = this.scripts
      .filter((script) => !query || [script.name, script.description || '', ...(script.tags || [])].join(' ').toLowerCase().includes(query))
      .sort((left, right) => {
        if (left.id === this.active?.id) return -1
        if (right.id === this.active?.id) return 1
        return left.name.localeCompare(right.name)
      })
    $('#script-count').textContent = String(this.scripts.length)
    $('#script-list').innerHTML = scripts.length
      ? scripts.map((script) => {
        const active = script.id === this.active?.id
        return `
          <article class="script-card${active ? ' active' : ''}" data-script-id="${escapeHtml(script.id)}">
            <div class="script-card-copy">
              <strong>${escapeHtml(script.name)}${active && this.dirty ? `<span class="script-dirty-badge">${escapeHtml(t('scripts.unsaved', {}, 'Unsaved changes'))}</span>` : ''}</strong>
              <span>${escapeHtml(script.description || t('scripts.storedDialogue', {}, 'Stored SSML-H dialogue'))}</span>
              <div class="script-card-metadata">
                <span>${escapeHtml(t('scripts.turnCount', { count: script.turns }, `${script.turns} turns`))}</span>
                <span>${escapeHtml(t('scripts.wordCount', { count: script.words }, `${script.words} words`))}</span>
                ${(script.tags || []).map((tag) => `<span>#${escapeHtml(tag)}</span>`).join('')}
              </div>
            </div>
            <div class="script-card-actions">
              <button class="secondary-button" type="button" data-script-action="use" ${active ? 'disabled' : ''}>${active ? `<i class="icon-check"></i><span>${escapeHtml(t('scripts.selected', {}, 'Selected'))}</span>` : `<i class="icon-book-open"></i><span>${escapeHtml(t('scripts.use', {}, 'Use script'))}</span>`}</button>
              <button class="icon-button bordered" type="button" data-script-action="edit" title="${escapeHtml(t('scripts.editNamed', { name: script.name }, `Edit ${script.name}`))}" aria-label="${escapeHtml(t('scripts.editNamed', { name: script.name }, `Edit ${script.name}`))}"><i class="icon-sliders-horizontal"></i></button>
              <button class="icon-button bordered danger-icon" type="button" data-script-action="delete" title="${escapeHtml(t('common.delete', {}, 'Delete'))}" aria-label="${escapeHtml(t('scripts.deleteNamed', { name: script.name }, `Delete ${script.name}`))}"><i class="icon-x"></i></button>
            </div>
          </article>`
      }).join('')
      : `<div class="script-empty">${escapeHtml(t(query ? 'scripts.noMatches' : 'scripts.empty', {}, query ? 'No matching scripts.' : 'No saved scripts yet.'))}</div>`
    this.renderSaveState()
  }

  async refresh(selectedId = this.active?.id) {
    const payload = await this.fetchJson('/tts/dialogue-scripts')
    this.scripts = payload.data || []
    if (selectedId) this.active = this.scripts.find((script) => script.id === selectedId) || null
    this.render()
  }

  currentDocument() {
    if (this.getInputType() === 'ssml-h') {
      const dirty = this.dirty
      this.runWithoutDirty(() => this.magicEditor.loadSSMLH($('#text-input').value))
      this.dirty = dirty
    }
    return this.magicEditor.toSSMLH()
  }

  download() {
    try {
      const name = this.active?.id || normalizedName($('#script-name').value) || 'voxcpm-dialogue'
      downloadDocument(this.currentDocument(), `${name}.ssml`)
    } catch (error) {
      this.showToast(this.errorMessage(error))
    }
  }

  startNew() {
    this.active = null
    this.dirty = false
    $('#script-name').value = ''
    this.setInputType('magic', { syncSsmlH: false })
    this.runWithoutDirty(() => this.magicEditor.newDocument())
    this.onDocumentChange(this.magicEditor.toSSMLH())
    this.render()
    this.magicEditor.canvas.querySelector('textarea')?.focus()
  }

  async load(scriptId) {
    const record = await this.fetchJson(`/tts/dialogue-scripts/${encodeURIComponent(scriptId)}`)
    this.setInputType('magic', { syncSsmlH: false })
    this.runWithoutDirty(() => this.magicEditor.loadSSMLH(record.document))
    this.active = record
    this.dirty = false
    $('#script-name').value = record.name
    this.onDocumentChange(this.magicEditor.toSSMLH())
    this.render()
    this.showToast(t('scripts.loaded', { name: record.name }, `Loaded ${record.name}.`), 'success')
  }

  async persist({ overwrite = false } = {}) {
    const name = $('#script-name').value.trim()
    if (!normalizedName(name)) throw new Error(t('scripts.nameRequired', {}, 'Enter a script name before saving.'))
    const saved = await this.fetchJson(overwrite ? `/tts/dialogue-scripts/${encodeURIComponent(this.active.id)}` : '/tts/dialogue-scripts', {
      method: overwrite ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        document: this.currentDocument(),
        description: this.active?.description || '',
        tags: this.active?.tags || [],
      }),
    })
    this.active = saved
    this.dirty = false
    $('#script-name').value = saved.name
    await this.refresh(saved.id)
    this.showToast(t(overwrite ? 'scripts.updatedNamed' : 'scripts.savedNamed', { name: saved.name }, `${overwrite ? 'Updated' : 'Saved'} ${saved.name}.`), 'success')
  }

  async openEdit(script) {
    const record = await this.fetchJson(`/tts/dialogue-scripts/${encodeURIComponent(script.id)}`)
    this.editingDetails = record
    $('#edit-script-name').textContent = record.name
    $('#edit-script-tags').value = (record.tags || []).join(', ')
    $('#edit-script-description').value = record.description || ''
    $('#edit-script-dialog').showModal()
  }

  closeEdit() {
    $('#edit-script-dialog').close()
    this.editingDetails = null
  }

  async updateDetails() {
    if (!this.editingDetails) return
    const record = this.editingDetails
    const saved = await this.fetchJson(`/tts/dialogue-scripts/${encodeURIComponent(record.id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: record.name,
        document: record.document,
        description: $('#edit-script-description').value.trim(),
        tags: tags($('#edit-script-tags').value),
      }),
    })
    if (this.active?.id === saved.id) this.active = saved
    this.closeEdit()
    await this.refresh(this.active?.id)
    this.showToast(t('scripts.detailsUpdatedNamed', { name: saved.name }, `Updated ${saved.name} details.`), 'success')
  }

  async import(file) {
    if (!file) return
    if (file.size > 1_000_000) throw new Error(t('scripts.importSize', {}, 'SSML-H scripts must be 1 MB or smaller.'))
    const source = await file.text()
    this.active = null
    this.dirty = false
    this.setInputType('magic', { syncSsmlH: false })
    this.runWithoutDirty(() => this.magicEditor.loadSSMLH(source))
    $('#script-name').value = file.name.replace(/\.(ssml|xml)$/i, '').slice(0, 80)
    this.onDocumentChange(this.magicEditor.toSSMLH())
    this.render()
    this.showToast(t('scripts.imported', { name: file.name }, `Imported ${file.name}.`), 'success')
  }

  bindEvents() {
    $('#script-import').addEventListener('click', () => $('#script-import-input').click())
    $('#script-import-input').addEventListener('change', async (event) => {
      const file = event.target.files[0]
      event.target.value = ''
      try { await this.import(file) } catch (error) { this.showToast(this.errorMessage(error)) }
    })
    $('#script-download').addEventListener('click', () => this.download())
    $('#script-name').addEventListener('input', () => this.renderSaveState())
    $('#script-filter').addEventListener('input', () => this.render())
    $('#script-save').addEventListener('click', () => {
      if (normalizedName($('#script-name').value) === this.active?.id) {
        $('#update-script-name').textContent = this.active.name
        $('#update-script-dialog').showModal()
      } else {
        this.persist().catch((error) => this.showToast(this.errorMessage(error)))
      }
    })
    $('#script-list').addEventListener('click', (event) => {
      const action = event.target.closest('[data-script-action]')?.dataset.scriptAction
      const scriptId = event.target.closest('[data-script-id]')?.dataset.scriptId
      const script = this.scripts.find((item) => item.id === scriptId)
      if (!action || !script) return
      if (action === 'use') this.load(scriptId).catch((error) => this.showToast(this.errorMessage(error)))
      if (action === 'edit') this.openEdit(script).catch((error) => this.showToast(this.errorMessage(error)))
      if (action === 'delete') {
        this.pendingDelete = script
        $('#delete-script-name').textContent = script.name
        $('#delete-script-dialog').showModal()
      }
    })
    $('#edit-script-close').addEventListener('click', () => this.closeEdit())
    $('#edit-script-cancel').addEventListener('click', () => this.closeEdit())
    $('#edit-script-dialog').addEventListener('click', (event) => { if (event.target === event.currentTarget) this.closeEdit() })
    $('#edit-script-form').addEventListener('submit', (event) => {
      event.preventDefault()
      this.updateDetails().catch((error) => this.showToast(this.errorMessage(error)))
    })
    $('#update-script-close').addEventListener('click', () => $('#update-script-dialog').close())
    $('#update-script-cancel').addEventListener('click', () => $('#update-script-dialog').close())
    $('#update-script-confirm').addEventListener('click', async (event) => {
      const button = event.currentTarget
      button.disabled = true
      try {
        await this.persist({ overwrite: true })
        $('#update-script-dialog').close()
      } catch (error) {
        this.showToast(this.errorMessage(error))
      } finally {
        button.disabled = false
      }
    })
    $('#delete-script-close').addEventListener('click', () => $('#delete-script-dialog').close())
    $('#delete-script-cancel').addEventListener('click', () => $('#delete-script-dialog').close())
    $('#delete-script-confirm').addEventListener('click', async (event) => {
      if (!this.pendingDelete) return
      const script = this.pendingDelete
      const button = event.currentTarget
      button.disabled = true
      try {
        await this.fetchJson(`/tts/dialogue-scripts/${encodeURIComponent(script.id)}`, { method: 'DELETE' })
        $('#delete-script-dialog').close()
        this.pendingDelete = null
        if (this.active?.id === script.id) this.startNew()
        await this.refresh()
        this.showToast(t('scripts.deletedNamed', { name: script.name }, `Deleted ${script.name}.`), 'success')
      } catch (error) {
        this.showToast(this.errorMessage(error))
      } finally {
        button.disabled = false
      }
    })
  }
}
