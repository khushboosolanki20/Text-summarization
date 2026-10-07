/**
 * Lightweight local history stored in the browser (localStorage).
 * No database: history lives only in this browser and can be cleared at any
 * time. Every access is wrapped in try/catch because storage can be disabled
 * or full (e.g. private windows); the app works without it.
 */

const KEY = 'intellisum.history.v1'
const MAX_ENTRIES = 25

export function loadHistory() {
  try {
    const entries = JSON.parse(localStorage.getItem(KEY) || '[]')
    return Array.isArray(entries) ? entries : []
  } catch {
    return []
  }
}

function save(entries) {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries))
    return true
  } catch {
    return false
  }
}

/** Store the essentials of a result (not the full sentence list, which can be large). */
export function addToHistory(result, sourceLabel) {
  const entry = {
    id: crypto.randomUUID?.() ?? String(Date.now()),
    date: new Date().toISOString(),
    method: result.method,
    length: result.length,
    source: sourceLabel,
    summary: result.summary,
    stats: {
      original_word_count: result.original_word_count,
      summary_word_count: result.summary_word_count,
      compression_ratio: result.compression_ratio,
      processing_time: result.processing_time,
      rouge1: result.metrics.rouge1,
      rouge2: result.metrics.rouge2,
      rougeL: result.metrics.rougeL,
    },
  }
  save([entry, ...loadHistory()].slice(0, MAX_ENTRIES))
  return entry
}

export function removeFromHistory(id) {
  const entries = loadHistory().filter((e) => e.id !== id)
  save(entries)
  return entries
}

export function clearHistory() {
  save([])
  return []
}
