const bootstrap = window.__VOXCPMTTS_UI__ || {
  locale: 'en',
  defaultLocale: 'en',
  locales: [{ code: 'en', name: 'English', path: '/en', direction: 'ltr', browserLanguage: 'en-US' }],
  storageKey: 'voxcpmtts-ui-locale-v1',
}

let messages = bootstrap.messages || {}

const LANGUAGE_CODES = {
  arabic: 'ar', burmese: 'my', chinese: 'zh', danish: 'da', dutch: 'nl', english: 'en',
  finnish: 'fi', french: 'fr', german: 'de', greek: 'el', hebrew: 'he', hindi: 'hi',
  indonesian: 'id', italian: 'it', japanese: 'ja', khmer: 'km', korean: 'ko', lao: 'lo',
  malay: 'ms', norwegian: 'no', polish: 'pl', portuguese: 'pt', russian: 'ru', spanish: 'es',
  swahili: 'sw', swedish: 'sv', tagalog: 'tl', thai: 'th', turkish: 'tr', vietnamese: 'vi',
}

async function loadCatalog(locale) {
  const response = await fetch(`/static/locales/${encodeURIComponent(locale)}.json`)
  if (!response.ok) throw new Error(`Unable to load UI locale ${locale}.`)
  return response.json()
}

function interpolate(template, variables) {
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) => (
    Object.hasOwn(variables, name) ? String(variables[name]) : match
  ))
}

export function t(key, variables = {}, fallback = key) {
  const value = messages[key]
  return interpolate(typeof value === 'string' ? value : fallback, variables)
}

export function browserLanguage() {
  return bootstrap.locales.find((item) => item.code === bootstrap.locale)?.browserLanguage || bootstrap.locale
}

export function languageLabel(language) {
  if (!language) return ''
  const key = String(language).toLowerCase().replaceAll(' ', '_')
  try {
    return new Intl.DisplayNames([browserLanguage()], { type: 'language' }).of(LANGUAGE_CODES[key]) || language
  } catch {
    return t(`languages.${key}`, {}, language)
  }
}

export function applyTranslations(root = document) {
  root.querySelectorAll('[data-i18n]').forEach((node) => {
    node.textContent = t(node.dataset.i18n, {}, node.textContent)
  })
  for (const attribute of ['title', 'aria-label', 'placeholder', 'alt']) {
    const dataName = `i18n${attribute.split('-').map((part) => part[0].toUpperCase() + part.slice(1)).join('')}`
    root.querySelectorAll(`[data-i18n-${attribute}]`).forEach((node) => {
      node.setAttribute(attribute, t(node.dataset[dataName], {}, node.getAttribute(attribute) || ''))
    })
  }
  root.querySelectorAll('[data-i18n-value]').forEach((node) => {
    node.value = t(node.dataset.i18nValue, {}, node.value || '')
  })
}

function populateLocalePicker() {
  const picker = document.querySelector('#ui-locale')
  if (!picker) return
  picker.replaceChildren(...bootstrap.locales.map((locale) => {
    const option = document.createElement('option')
    option.value = locale.code
    option.textContent = locale.name
    option.selected = locale.code === bootstrap.locale
    return option
  }))
  picker.addEventListener('change', () => {
    const locale = bootstrap.locales.find((item) => item.code === picker.value)
    if (!locale) return
    try { localStorage.setItem(bootstrap.storageKey, locale.code) } catch {}
    window.location.assign(`${locale.path}${window.location.search}${window.location.hash}`)
  })
}

export async function initializeI18n() {
  if (!Object.keys(messages).length) {
    try {
      messages = await loadCatalog(bootstrap.locale)
    } catch (error) {
      console.error(error)
    }
  }

  document.documentElement.lang = bootstrap.locale
  document.documentElement.dir = bootstrap.locales.find((item) => item.code === bootstrap.locale)?.direction || 'ltr'
  document.title = t('app.title', {}, document.title)
  try { localStorage.setItem(bootstrap.storageKey, bootstrap.locale) } catch {}
  applyTranslations()
  populateLocalePicker()
}
