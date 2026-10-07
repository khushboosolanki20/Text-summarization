import axios from 'axios'

// In development Vite proxies /api to the FastAPI server (see vite.config.js).
// For a production build, set VITE_API_BASE_URL to the backend origin.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 2 * 60 * 1000, // starting a job uploads the file; polling requests are tiny
})

/**
 * Convert any Axios error into a short, human-readable message.
 * The backend always returns `{ detail: string }` for handled errors.
 */
export function toErrorMessage(error) {
  if (error?.isJobError) return error.message
  if (error.response?.data?.detail) {
    const { detail } = error.response.data
    return typeof detail === 'string' ? detail : 'The request was invalid.'
  }
  if (error.code === 'ECONNABORTED') return 'The request timed out.'
  if (error.request) return 'Cannot reach the IntelliSum server. Is the backend running?'
  return 'Something went wrong. Please try again.'
}

export async function getHealth() {
  const { data } = await api.get('/api/health')
  return data
}

export async function getConfig() {
  const { data } = await api.get('/api/config')
  return data
}

/** Ask the server to start loading BART in the background (fire and forget). */
export function warmup() {
  return api.post('/api/warmup').catch(() => null)
}

/** Start a summarization job for pasted text. Returns the job object. */
export async function startTextJob({ text, method, length, referenceSummary }) {
  const { data } = await api.post('/api/summarize/text/async', {
    text,
    method,
    length,
    reference_summary: referenceSummary?.trim() || null,
  })
  return data
}

/** Start a summarization job for an uploaded file. Returns the job object. */
export async function startFileJob({ file, method, length, referenceSummary }) {
  const form = new FormData()
  form.append('file', file)
  form.append('method', method)
  form.append('length', length)
  if (referenceSummary?.trim()) form.append('reference_summary', referenceSummary.trim())
  const { data } = await api.post('/api/summarize/file/async', form)
  return data
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * Poll a job until it completes. Calls onProgress(job.progress) on every
 * update and resolves with the summarization result. Rejects with a readable
 * message if the job fails; stops quietly when `signal` is aborted.
 */
export async function waitForJob(jobId, { onProgress, signal, interval = 700 } = {}) {
  for (;;) {
    if (signal?.aborted) throw Object.assign(new Error('Cancelled'), { cancelled: true })
    const { data: job } = await api.get(`/api/jobs/${jobId}`, { signal })
    if (job.progress) onProgress?.(job.progress)
    if (job.status === 'completed') return job.result
    if (job.status === 'failed') throw Object.assign(new Error(job.error), { isJobError: true })
    await sleep(interval)
  }
}

/** ROUGE (and optionally faithfulness) for a summary against a reference. */
export async function evaluateSummary({ summary, referenceSummary, originalText }) {
  const { data } = await api.post('/api/evaluate', {
    summary,
    reference_summary: referenceSummary,
    original_text: originalText ?? null,
  })
  return data
}

export default api
