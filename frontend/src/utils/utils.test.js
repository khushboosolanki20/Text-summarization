import { describe, expect, it, vi } from 'vitest'
import { summaryFilename } from './download'
import { countWords, formatBytes, formatNumber, formatPercent, formatScore, formatSeconds } from './format'
import { addToHistory, clearHistory, loadHistory, removeFromHistory } from './history'
import { validateFile } from './validation'

const CONFIG = { supported_file_types: ['.txt', '.pdf', '.docx'], max_upload_mb: 1 }
const file = (name, size) => new File([new Uint8Array(size)], name)

describe('format', () => {
  it('formats numbers, percentages and ROUGE scores', () => {
    expect(formatNumber(4532)).toBe('4,532')
    expect(formatNumber(null)).toBe('—')
    expect(formatPercent(93.678)).toBe('93.7%')
    expect(formatScore(0.6463)).toBe('64.6') // API gives 0–1; shown like papers, 0–100
    expect(formatScore(null)).toBe('—')
  })

  it('formats durations across scales', () => {
    expect(formatSeconds(0.042)).toBe('42 ms')
    expect(formatSeconds(19.34)).toBe('19.3 s')
    expect(formatSeconds(125)).toBe('2 min 5 s')
  })

  it('counts words and sizes', () => {
    expect(countWords('  one two\nthree  ')).toBe(3)
    expect(countWords('   ')).toBe(0)
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2 KB')
    expect(formatBytes(3 * 1024 * 1024)).toBe('3.0 MB')
  })

  it('builds safe download filenames', () => {
    expect(summaryFilename('Annual Report (2025).pdf', 'hybrid')).toBe('Annual_Report_2025__hybrid_summary.txt')
    expect(summaryFilename(undefined, 'tfidf')).toBe('text_tfidf_summary.txt')
  })
})

describe('validateFile', () => {
  it('accepts supported files within the size limit', () => {
    expect(validateFile(file('report.PDF', 100), CONFIG)).toBeNull()
  })

  it.each([
    ['photo.png', 10, 'Unsupported file type'],
    ['old.doc', 10, '.docx'],
    ['empty.txt', 0, 'empty'],
    ['huge.pdf', 2 * 1024 * 1024, 'too large'],
  ])('rejects %s', (name, size, message) => {
    expect(validateFile(file(name, size), CONFIG)).toContain(message)
  })
})

describe('history', () => {
  const result = (summary) => ({
    method: 'tfidf',
    length: 'short',
    summary,
    original_word_count: 100,
    summary_word_count: 20,
    compression_ratio: 80,
    processing_time: 0.1,
    metrics: { rouge1: null, rouge2: null, rougeL: null },
  })

  it('stores newest first with essential fields only', () => {
    addToHistory(result('first'), 'a.txt')
    addToHistory(result('second'), 'Pasted text')
    const entries = loadHistory()
    expect(entries.map((e) => e.summary)).toEqual(['second', 'first'])
    expect(entries[0]).toMatchObject({ method: 'tfidf', source: 'Pasted text', stats: { compression_ratio: 80 } })
    expect(entries[0].sentences).toBeUndefined()
  })

  it('keeps at most 25 entries', () => {
    for (let i = 0; i < 30; i++) addToHistory(result(`s${i}`), 'x')
    expect(loadHistory()).toHaveLength(25)
    expect(loadHistory()[0].summary).toBe('s29')
  })

  it('removes and clears entries', () => {
    const entry = addToHistory(result('one'), 'x')
    addToHistory(result('two'), 'x')
    expect(removeFromHistory(entry.id).map((e) => e.summary)).toEqual(['two'])
    expect(clearHistory()).toEqual([])
    expect(loadHistory()).toEqual([])
  })

  it('survives broken or unavailable storage', () => {
    localStorage.setItem('intellisum.history.v1', '{not json')
    expect(loadHistory()).toEqual([])
    const spy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceeded')
    })
    expect(() => addToHistory(result('x'), 'x')).not.toThrow()
    spy.mockRestore()
  })
})
