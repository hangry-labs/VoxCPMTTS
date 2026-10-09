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

export function serializeMagicDocument(blocks) {
  const body = blocks.map((block) => {
    if (block.type === 'break') return `  <break time="${Math.round(block.milliseconds)}ms"/>`
    const text = block.text.trim()
    if (!text) return ''
    if (!block.voice) return `  ${escapeXml(text)}`
    const direction = block.direction ? ` h:direction="${escapeXml(block.direction)}"` : ''
    return `  <voice name="${escapeXml(block.voice)}"${direction}>${escapeXml(text)}</voice>`
  }).filter(Boolean).join('\n')
  return `<speak version="1.1" xmlns="${SSML_NAMESPACE}" xmlns:h="${SSML_H_NAMESPACE}">\n${body}\n</speak>`
}
