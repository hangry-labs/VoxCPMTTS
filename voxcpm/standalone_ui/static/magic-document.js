const SSML_NAMESPACE = 'http://www.w3.org/2001/10/synthesis'
const SSML_H_NAMESPACE = 'https://hangrylabs.app/ns/ssml-h/1.0'

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}

export function serializeMagicDocument(blocks, options = {}) {
  const body = blocks.map((block) => {
    if (block.type === 'break') return `  <break time="${Math.round(block.milliseconds)}ms"/>`
    const text = block.text.trim()
    if (!text) return ''
    if (!block.voice) return `  ${escapeXml(text)}`
    const direction = block.direction ? ` h:direction="${escapeXml(block.direction)}"` : ''
    return `  <voice name="${escapeXml(block.voice)}"${direction}>${escapeXml(text)}</voice>`
  }).filter(Boolean)
  const metadata = String(options.metadataXml || '').trim()
  if (metadata) body.unshift(...metadata.split('\n').map((line) => `  ${line.trim()}`))
  const language = options.language ? ` xml:lang="${escapeXml(options.language)}"` : ''
  return `<speak version="1.1" xmlns="${SSML_NAMESPACE}" xmlns:h="${SSML_H_NAMESPACE}"${language}>\n${body.join('\n')}\n</speak>`
}

function parseBreakTime(value) {
  const match = String(value || '').trim().match(/^(\d+(?:\.\d+)?)(ms|s)$/i)
  if (!match) throw new Error('Magic supports break times written in milliseconds or seconds.')
  const milliseconds = Number(match[1]) * (match[2].toLowerCase() === 's' ? 1000 : 1)
  if (!Number.isFinite(milliseconds) || milliseconds < 0 || milliseconds > 10_000) {
    throw new Error('Magic supports pauses between 0 and 10000 milliseconds.')
  }
  return Math.round(milliseconds)
}

function speechBlock(element) {
  if ([...element.children].length) {
    throw new Error('Magic import supports plain text inside each voice turn. Keep nested SSML in the SSML-H editor.')
  }
  return {
    type: 'speech',
    text: element.textContent.trim(),
    voice: element.getAttribute('name') || '',
    direction: element.getAttributeNS(SSML_H_NAMESPACE, 'direction') || element.getAttribute('h:direction') || '',
  }
}

export function parseMagicDocument(source) {
  const parser = new DOMParser()
  const document = parser.parseFromString(String(source || ''), 'application/xml')
  const parseError = document.querySelector('parsererror')
  if (parseError) throw new Error(`Invalid SSML-H: ${parseError.textContent.trim()}`)
  const root = document.documentElement
  if (root.localName !== 'speak' || root.namespaceURI !== SSML_NAMESPACE) {
    throw new Error('SSML-H must use a namespaced <speak> root element.')
  }

  const blocks = []
  let metadataXml = ''
  for (const node of root.childNodes) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent.trim()
      if (text) blocks.push({ type: 'speech', text, voice: '', direction: '' })
      continue
    }
    if (node.nodeType !== Node.ELEMENT_NODE) continue
    if (node.namespaceURI !== SSML_NAMESPACE) {
      throw new Error(`Magic cannot edit the <${node.localName}> element at the document level.`)
    }
    if (node.localName === 'metadata') {
      if (metadataXml) throw new Error('Magic supports one metadata section.')
      metadataXml = new XMLSerializer().serializeToString(node)
      continue
    }
    if (node.localName === 'voice') {
      blocks.push(speechBlock(node))
      continue
    }
    if (node.localName === 'break') {
      blocks.push({ type: 'break', milliseconds: parseBreakTime(node.getAttribute('time')) })
      continue
    }
    throw new Error(`Magic cannot edit <${node.localName}> blocks. Keep this document in the SSML-H editor.`)
  }
  if (!blocks.length) blocks.push({ type: 'speech', text: '', voice: '', direction: '' })
  return {
    blocks,
    metadataXml,
    language: root.getAttributeNS('http://www.w3.org/XML/1998/namespace', 'lang') || '',
  }
}
