import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ABSTRACTIVE_RESULT, CONFIG, EXTRACTIVE_RESULT, LONG_TEXT } from '../test/fixtures'
import OptionGroup from './input/OptionGroup'
import SummarizeForm from './input/SummarizeForm'
import ResultView from './results/ResultView'

vi.mock('../services/api', async (importOriginal) => ({
  ...(await importOriginal()),
  warmup: vi.fn(() => Promise.resolve()),
  evaluateSummary: vi.fn(),
}))
const api = await import('../services/api')

beforeEach(() => vi.clearAllMocks())

// ---------------------------------------------------------------- the form

describe('SummarizeForm', () => {
  const setup = () => {
    const onSubmit = vi.fn()
    render(<SummarizeForm config={CONFIG} running={false} onSubmit={onSubmit} />)
    return { onSubmit, user: userEvent.setup() }
  }

  it('keeps Generate disabled until there is enough text', async () => {
    const { user } = setup()
    const button = screen.getByRole('button', { name: /generate summary/i })
    expect(button).toBeDisabled()

    await user.type(screen.getByLabelText('Text to summarize'), 'Too short to summarize.')
    expect(screen.getByText(/at least 40 words needed/)).toBeInTheDocument()
    expect(button).toBeDisabled()

    fireEvent.change(screen.getByLabelText('Text to summarize'), { target: { value: LONG_TEXT } })
    expect(button).toBeEnabled()
  })

  it('submits the chosen method, length and reference', async () => {
    const { onSubmit, user } = setup()
    fireEvent.change(screen.getByLabelText('Text to summarize'), { target: { value: LONG_TEXT } })
    await user.click(screen.getByRole('radio', { name: /TextRank/ }))
    await user.click(screen.getByRole('radio', { name: /Long/ }))
    await user.click(screen.getByText('Reference summary'))
    await user.type(screen.getByPlaceholderText(/human-written summary/), 'Solar grew.')
    await user.click(screen.getByRole('button', { name: /generate summary/i }))

    expect(onSubmit).toHaveBeenCalledWith({
      text: LONG_TEXT,
      file: undefined,
      method: 'textrank',
      length: 'long',
      referenceSummary: 'Solar grew.',
    })
  })

  it('starts loading BART when an abstractive method is chosen', async () => {
    const { user } = setup() // default method is Hybrid
    expect(api.warmup).toHaveBeenCalledTimes(1)
    await user.click(screen.getByRole('radio', { name: /TF-IDF/ }))
    expect(api.warmup).toHaveBeenCalledTimes(1) // extractive: no warm-up
    await user.click(screen.getByRole('radio', { name: /^BART/ }))
    expect(api.warmup).toHaveBeenCalledTimes(2)
  })

  it('rejects unsupported uploads before sending them', async () => {
    const { user } = setup()
    await user.click(screen.getByRole('tab', { name: /upload file/i }))
    const input = document.querySelector('input[type="file"]')
    fireEvent.change(input, { target: { files: [new File(['x'], 'photo.png')] } })
    expect(screen.getByRole('alert')).toHaveTextContent(/Unsupported file type ".png"/)
    expect(screen.getByRole('button', { name: /generate summary/i })).toBeDisabled()
  })

  it('accepts a valid file and submits it', async () => {
    const { onSubmit, user } = setup()
    await user.click(screen.getByRole('tab', { name: /upload file/i }))
    const pdf = new File(['%PDF-1.7'], 'report.pdf')
    fireEvent.change(document.querySelector('input[type="file"]'), { target: { files: [pdf] } })
    expect(screen.getByText('report.pdf')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /generate summary/i }))
    expect(onSubmit.mock.calls[0][0]).toMatchObject({ file: pdf, text: undefined, method: 'hybrid' })
  })
})

describe('OptionGroup', () => {
  it('supports arrow-key navigation like a native radio group', () => {
    const onChange = vi.fn()
    const options = [
      { value: 'short', label: 'Short' },
      { value: 'medium', label: 'Medium' },
      { value: 'long', label: 'Long' },
    ]
    render(<OptionGroup label="Summary length" options={options} value="long" onChange={onChange} />)
    const group = screen.getByRole('radiogroup', { name: 'Summary length' })
    expect(screen.getByRole('radio', { name: 'Long' })).toHaveAttribute('aria-checked', 'true')
    fireEvent.keyDown(group, { key: 'ArrowRight' })
    expect(onChange).toHaveBeenLastCalledWith('short') // wraps around
    fireEvent.keyDown(group, { key: 'ArrowLeft' })
    expect(onChange).toHaveBeenLastCalledWith('medium')
  })
})

// ------------------------------------------------------------- the results

describe('ResultView (extractive)', () => {
  it('shows the comparison, summary and selected sentences', () => {
    render(<ResultView result={EXTRACTIVE_RESULT} onNew={() => {}} />)
    expect(screen.getByText('50.0%')).toBeInTheDocument() // compression
    expect(screen.getByText('2 / 4')).toBeInTheDocument() // sentences selected
    expect(screen.getByText(EXTRACTIVE_RESULT.summary)).toBeInTheDocument()
    expect(screen.getByText('Why these sentences?')).toBeInTheDocument()
    expect(screen.getAllByText('p. 2').length).toBeGreaterThan(0) // page numbers shown
  })

  it('explains that ROUGE needs a reference and evaluates one on request', async () => {
    api.evaluateSummary.mockResolvedValue({ rouge: ABSTRACTIVE_RESULT.metrics.rouge })
    render(<ResultView result={EXTRACTIVE_RESULT} onNew={() => {}} />)
    expect(screen.getByText('ROUGE not calculated')).toBeInTheDocument()

    const user = userEvent.setup()
    await user.type(screen.getByLabelText('Reference summary'), 'Solar grew fastest.')
    await user.click(screen.getByRole('button', { name: 'Calculate ROUGE' }))

    expect(api.evaluateSummary).toHaveBeenCalledWith({ summary: EXTRACTIVE_RESULT.summary, referenceSummary: 'Solar grew fastest.' })
    const table = await screen.findByRole('table')
    expect(within(table).getByText('ROUGE-1')).toBeInTheDocument()
    expect(within(table).getByText('50.0')).toBeInTheDocument() // F1 0.5 shown as 50.0
  })

  it('does not show a faithfulness panel for copied sentences', () => {
    render(<ResultView result={EXTRACTIVE_RESULT} onNew={() => {}} />)
    expect(screen.queryByText(/Potentially unsupported content/)).not.toBeInTheDocument()
  })
})

describe('ResultView (abstractive)', () => {
  it('shows ROUGE scores in a table', () => {
    render(<ResultView result={ABSTRACTIVE_RESULT} onNew={() => {}} />)
    const row = within(screen.getByRole('table')).getByText('ROUGE-Lsum').closest('tr')
    expect(row).toHaveTextContent('55.0') // precision 0.55
    expect(row).toHaveTextContent('38.0') // recall 0.38
    expect(row).toHaveTextContent('45.0') // F1 0.45
  })

  it('flags potentially unsupported sentences with reasons and the source passage', () => {
    render(<ResultView result={ABSTRACTIVE_RESULT} onNew={() => {}} />)
    expect(screen.getByText('Experimental')).toBeInTheDocument()
    expect(screen.getByText(/1 of 2 summary sentences/)).toBeInTheDocument()
    expect(screen.getByText('number(s) not found in the source: 15')).toBeInTheDocument()
    expect(screen.getByText(ABSTRACTIVE_RESULT.sentences[2].text)).toBeInTheDocument() // closest passage
    // The flagged sentence is also marked inside the summary itself.
    expect(screen.getByTitle(/Potentially unsupported: number/)).toHaveTextContent('Installations rose by 15 percent.')
  })

  it('shows model details and no sentence-selection panel', () => {
    render(<ResultView result={ABSTRACTIVE_RESULT} onNew={() => {}} />)
    expect(screen.getByText('facebook/bart-large-cnn')).toBeInTheDocument()
    expect(screen.getByText('CPU')).toBeInTheDocument()
    expect(screen.queryByText('Why these sentences?')).not.toBeInTheDocument()
  })

  it('shows document warnings', () => {
    render(<ResultView result={{ ...ABSTRACTIVE_RESULT, warnings: ['2 of 10 pages were skipped'] }} onNew={() => {}} />)
    expect(screen.getByText('2 of 10 pages were skipped')).toBeInTheDocument()
  })
})

describe('copy and download', () => {
  it('copies the summary to the clipboard', async () => {
    const writeText = vi.fn(() => Promise.resolve())
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    render(<ResultView result={EXTRACTIVE_RESULT} onNew={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Copy' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument())
    expect(writeText).toHaveBeenCalledWith(EXTRACTIVE_RESULT.summary)
  })
})
