import { useEffect, useState } from 'react'
import { warmup } from '../../services/api'
import { countWords, formatNumber } from '../../utils/format'
import Icon from '../ui/Icon'
import { Banner, Button, Card } from '../ui/primitives'
import FileDropzone from './FileDropzone'
import OptionGroup from './OptionGroup'

const TYPE_TAGS = { extractive: 'Extractive', abstractive: 'Abstractive', hybrid: 'Ext. → Abs.' }

export default function SummarizeForm({ config, running, onSubmit }) {
  const [source, setSource] = useState('text') // "text" | "file"
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [fileError, setFileError] = useState(null)
  const [method, setMethod] = useState('hybrid')
  const [length, setLength] = useState('medium')
  const [reference, setReference] = useState('')

  // Start loading BART in the background as soon as an abstractive method is
  // chosen, so the first summary doesn't wait for the model.
  useEffect(() => {
    if (method === 'bart' || method === 'hybrid') warmup()
  }, [method])

  const words = countWords(text)
  const tooShort = source === 'text' && words > 0 && words < config.min_input_words
  const tooLongForBart = source === 'text' && method === 'bart' && words > config.abstractive_max_input_words
  const ready = source === 'text' ? words >= config.min_input_words && !tooLongForBart : Boolean(file)

  const submit = (e) => {
    e.preventDefault()
    if (!ready || running) return
    onSubmit({
      text: source === 'text' ? text : undefined,
      file: source === 'file' ? file : undefined,
      method,
      length,
      referenceSummary: reference,
    })
  }

  const methodOptions = config.methods.map((m) => ({
    value: m.id,
    label: m.name,
    tag: TYPE_TAGS[m.type],
    description: m.description,
  }))
  const lengthOptions = Object.entries(config.lengths).map(([id, ratio]) => ({
    value: id,
    label: id[0].toUpperCase() + id.slice(1),
    hint: `~${Math.round(ratio * 100)}%`,
  }))

  return (
    <form onSubmit={submit} className="grid gap-6 lg:grid-cols-5">
      <Card title="1 · Your document" className="lg:col-span-3">
        <div role="tablist" aria-label="Document source" className="mb-4 inline-flex rounded-xl bg-slate-100 p-1">
          {[
            { id: 'text', label: 'Paste text', icon: 'text' },
            { id: 'file', label: 'Upload file', icon: 'upload' },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={source === tab.id}
              onClick={() => setSource(tab.id)}
              disabled={running}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition ${
                source === tab.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Icon name={tab.icon} className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {source === 'text' ? (
          <div>
            <label htmlFor="document-text" className="sr-only">
              Text to summarize
            </label>
            <textarea
              id="document-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={running}
              placeholder="Paste an article, report or essay here…"
              className="h-56 w-full resize-y rounded-xl border border-slate-300 bg-white p-4 text-sm leading-relaxed text-slate-800 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className={tooShort ? 'font-medium text-amber-700' : 'text-slate-500'}>
                {formatNumber(words)} words
                {tooShort && ` · at least ${config.min_input_words} words needed`}
                {tooLongForBart &&
                  ` · BART accepts up to ${formatNumber(config.abstractive_max_input_words)} words; use Hybrid for longer texts`}
              </span>
              {text && (
                <button type="button" onClick={() => setText('')} disabled={running} className="text-slate-500 hover:text-slate-800">
                  Clear
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <FileDropzone file={file} onFile={setFile} onError={setFileError} config={config} disabled={running} />
            {fileError && <Banner kind="error" title="This file can't be used">{fileError}</Banner>}
          </div>
        )}
      </Card>

      <Card title="2 · Options" className="lg:col-span-2">
        <div className="space-y-5">
          <OptionGroup
            label="Summarization method"
            variant="cards"
            options={methodOptions}
            value={method}
            onChange={setMethod}
            disabled={running}
          />
          <OptionGroup label="Summary length" options={lengthOptions} value={length} onChange={setLength} disabled={running} />

          <details className="group rounded-xl border border-slate-200">
            <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-sm font-medium text-slate-700">
              <Icon name="chevron" className="h-4 w-4 shrink-0 transition group-open:rotate-90" />
              <span>
                Reference summary <span className="font-normal text-slate-500">(optional, for ROUGE)</span>
              </span>
            </summary>
            <div className="border-t border-slate-100 p-3">
              <textarea
                value={reference}
                onChange={(e) => setReference(e.target.value)}
                disabled={running}
                rows={3}
                placeholder="A human-written summary to compare against…"
                className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
              <p className="mt-1 text-xs text-slate-500">
                ROUGE scores need a reference. Without one they are not calculated.
              </p>
            </div>
          </details>

          <Button type="submit" variant="primary" size="lg" icon="sparkles" disabled={!ready || running} className="w-full">
            {running ? 'Summarizing…' : 'Generate summary'}
          </Button>
        </div>
      </Card>
    </form>
  )
}
