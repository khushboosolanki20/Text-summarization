export const formatNumber = (n) => (n == null ? '—' : new Intl.NumberFormat('en-US').format(n))

export const formatPercent = (value, digits = 1) => (value == null ? '—' : `${value.toFixed(digits)}%`)

/** ROUGE scores are 0–1 from the API; show them as percentages like papers do. */
export const formatScore = (value) => (value == null ? '—' : (value * 100).toFixed(1))

export function formatSeconds(seconds) {
  if (seconds == null) return '—'
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes} min ${Math.round(seconds % 60)} s`
}

export function formatDate(iso) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export const countWords = (text) => (text.trim() ? text.trim().split(/\s+/).length : 0)

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export const ROUGE_NAMES = { rouge1: 'ROUGE-1', rouge2: 'ROUGE-2', rougeL: 'ROUGE-L', rougeLsum: 'ROUGE-Lsum' }

export const METHOD_LABELS ={ tfidf: 'TF-IDF', textrank: 'TextRank', bart: 'BART', hybrid: 'Hybrid' }

export const STRATEGY_LABELS = {
  extractive: 'Extractive selection',
  single_pass: 'Single BART pass',
  fused: 'Chunked, then fused in a final pass',
  concatenated: 'Section-by-section (chunked)',
  max_levels_reached: 'Chunked (maximum rounds reached)',
}

/** Workflow node ids -> short labels for the process view. */
export const NODE_LABELS = {
  preprocess: 'Preprocess',
  extractive_summarize: 'Extractive scoring',
  select_key_sentences: 'TextRank selection',
  check_length: 'Check length',
  abstractive_single_pass: 'BART generation',
  chunk_document: 'Chunk document',
  summarize_chunks: 'Summarize chunks',
  combine_summaries: 'Combine summaries',
  final_summarization: 'Final pass',
  postprocess: 'Post-process',
  evaluate: 'Evaluate',
}
