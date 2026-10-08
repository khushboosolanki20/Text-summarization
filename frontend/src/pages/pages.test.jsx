import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { CONFIG, EXTRACTIVE_RESULT, LONG_TEXT } from '../test/fixtures'
import { addToHistory } from '../utils/history'
import HistoryPage from './HistoryPage'
import SummarizePage from './SummarizePage'

vi.mock('../services/api', async (importOriginal) => ({
  ...(await importOriginal()),
  getConfig: vi.fn(),
  warmup: vi.fn(() => Promise.resolve()),
  startTextJob: vi.fn(),
  waitForJob: vi.fn(),
}))
const api = await import('../services/api')

beforeEach(() => {
  vi.clearAllMocks()
  api.getConfig.mockResolvedValue(CONFIG)
  api.startTextJob.mockResolvedValue({ job_id: 'job-1', status: 'queued' })
  Element.prototype.scrollIntoView = vi.fn()
})

const renderPage = (page) => render(<MemoryRouter>{page}</MemoryRouter>)

async function submitText(user) {
  fireEvent.change(screen.getByLabelText('Text to summarize'), { target: { value: LONG_TEXT } })
  await user.click(screen.getByRole('radio', { name: /TextRank/ }))
  await user.click(screen.getByRole('button', { name: /generate summary/i }))
}

describe('SummarizePage flow', () => {
  it('starts a job, shows progress, then the result, and saves it to history', async () => {
    let finish
    api.waitForJob.mockImplementation((jobId, { onProgress }) => {
      onProgress({ stage: 'Scoring and selecting sentences', done: 0, total: 1 })
      return new Promise((resolve) => {
        finish = resolve
      })
    })
    const user = userEvent.setup()
    renderPage(<SummarizePage />)
    expect(screen.getByText('Your summary will appear here')).toBeInTheDocument()

    await submitText(user)
    expect(api.startTextJob).toHaveBeenCalledWith({ text: LONG_TEXT, method: 'textrank', length: 'medium', referenceSummary: '' })
    expect(await screen.findByText('Scoring and selecting sentences')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /summarizing/i })).toBeDisabled()

    finish(EXTRACTIVE_RESULT)
    // The results view is lazy-loaded (code splitting), so allow time for the import.
    expect(await screen.findByText(EXTRACTIVE_RESULT.summary, {}, { timeout: 15000 })).toBeInTheDocument()
    expect(JSON.parse(localStorage.getItem('intellisum.history.v1'))[0].summary).toBe(EXTRACTIVE_RESULT.summary)
  }, 30000) // the lazy import can be slow on a busy machine; the default test limit is 5 s

  it('shows a readable error when the job fails', async () => {
    api.waitForJob.mockRejectedValue(Object.assign(new Error('The text is too short to summarize.'), { isJobError: true }))
    const user = userEvent.setup()
    renderPage(<SummarizePage />)
    await submitText(user)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The summary could not be created')
    expect(alert).toHaveTextContent('The text is too short to summarize.')
  })

  it('cancel returns to the empty state', async () => {
    api.waitForJob.mockImplementation(() => new Promise(() => {})) // never finishes
    const user = userEvent.setup()
    renderPage(<SummarizePage />)
    await submitText(user)
    await user.click(await screen.findByRole('button', { name: 'Cancel' }))
    expect(screen.getByText('Your summary will appear here')).toBeInTheDocument()
  })
})

describe('HistoryPage', () => {
  it('shows an empty state without history', () => {
    renderPage(<HistoryPage />)
    expect(screen.getByText('No summaries yet')).toBeInTheDocument()
  })

  it('lists saved summaries and deletes them', async () => {
    addToHistory(EXTRACTIVE_RESULT, 'report.pdf')
    const user = userEvent.setup()
    renderPage(<HistoryPage />)
    expect(screen.getByText('report.pdf')).toBeInTheDocument()
    expect(screen.getByText(EXTRACTIVE_RESULT.summary)).toBeInTheDocument()
    expect(screen.getByText('44 → 22')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Delete from history' }))
    expect(screen.getByText('No summaries yet')).toBeInTheDocument()
  })
})
