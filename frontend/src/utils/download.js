/** Save text as a .txt file in the browser. */
export function downloadText(text, filename = 'summary.txt') {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

/** Copy text to the clipboard; resolves true on success. */
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

export function summaryFilename(source, method) {
  const base = (source || 'text').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 40)
  return `${base}_${method}_summary.txt`
}
