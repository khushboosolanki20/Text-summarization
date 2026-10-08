import { afterEach, describe, expect, it, vi } from 'vitest'
import api, { toErrorMessage, waitForJob } from './api'

afterEach(() => vi.restoreAllMocks())

describe('toErrorMessage', () => {
  it('uses the backend detail message', () => {
    const error = { response: { data: { detail: 'The text is too short to summarize.' } } }
    expect(toErrorMessage(error)).toBe('The text is too short to summarize.')
  })

  it('never shows non-string details', () => {
    expect(toErrorMessage({ response: { data: { detail: [{ loc: ['body'] }] } } })).toBe('The request was invalid.')
  })

  it('explains network problems in plain language', () => {
    expect(toErrorMessage({ request: {} })).toMatch(/Cannot reach the IntelliSum server/)
    expect(toErrorMessage({ code: 'ECONNABORTED', request: {} })).toBe('The request timed out.')
    expect(toErrorMessage({})).toBe('Something went wrong. Please try again.')
  })

  it('passes job failure messages through', () => {
    expect(toErrorMessage(Object.assign(new Error('OCR is not supported'), { isJobError: true }))).toBe('OCR is not supported')
  })
})

describe('waitForJob', () => {
  const responses = (...jobs) => {
    const get = vi.spyOn(api, 'get')
    jobs.forEach((job) => get.mockResolvedValueOnce({ data: job }))
    return get
  }

  it('polls until completed and reports progress', async () => {
    const get = responses(
      { status: 'running', progress: { stage: 'Summarizing chunks', done: 1, total: 3 } },
      { status: 'running', progress: { stage: 'Summarizing chunks', done: 2, total: 3 } },
      { status: 'completed', progress: { stage: 'Completed', done: 1, total: 1 }, result: { summary: 'Done.' } },
    )
    const progress = []
    const result = await waitForJob('abc', { interval: 0, onProgress: (p) => progress.push(p.done) })
    expect(result).toEqual({ summary: 'Done.' })
    expect(progress).toEqual([1, 2, 1])
    expect(get).toHaveBeenCalledTimes(3)
    expect(get.mock.calls[0][0]).toBe('/api/jobs/abc')
  })

  it('rejects with the job error message when the job fails', async () => {
    responses({ status: 'failed', error: 'The PDF is password-protected.' })
    const error = await waitForJob('abc', { interval: 0 }).catch((e) => e)
    expect(error.isJobError).toBe(true)
    expect(toErrorMessage(error)).toBe('The PDF is password-protected.')
  })

  it('stops when cancelled', async () => {
    const controller = new AbortController()
    controller.abort()
    const error = await waitForJob('abc', { signal: controller.signal }).catch((e) => e)
    expect(error.cancelled).toBe(true)
  })
})
