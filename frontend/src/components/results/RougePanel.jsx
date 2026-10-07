import { useState } from 'react'
import { evaluateSummary, toErrorMessage } from '../../services/api'
import { ROUGE_NAMES, formatScore } from '../../utils/format'
import { Banner, Button, Card } from '../ui/primitives'
import { RougeChart } from './charts'

/** Table view of every ROUGE value (the chart's accessible equivalent). */
function RougeTable({ rouge }) {
  return (
    <table className="mt-4 w-full text-sm">
      <caption className="sr-only">ROUGE scores (0–100)</caption>
      <thead>
        <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
          <th className="py-2 font-medium">Metric</th>
          <th className="py-2 text-right font-medium">Precision</th>
          <th className="py-2 text-right font-medium">Recall</th>
          <th className="py-2 text-right font-medium">F1</th>
        </tr>
      </thead>
      <tbody className="tabular-nums">
        {Object.entries(ROUGE_NAMES).map(([key, name]) => (
          <tr key={key} className="border-b border-slate-100 last:border-0">
            <td className="py-2 font-medium text-slate-800">{name}</td>
            <td className="py-2 text-right">{formatScore(rouge[key].precision)}</td>
            <td className="py-2 text-right">{formatScore(rouge[key].recall)}</td>
            <td className="py-2 text-right font-semibold text-slate-900">{formatScore(rouge[key].f1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function RougePanel({ result }) {
  const [rouge, setRouge] = useState(result.metrics.rouge)
  const [reference, setReference] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const evaluate = async () => {
    setBusy(true)
    setError(null)
    try {
      const data = await evaluateSummary({ summary: result.summary, referenceSummary: reference })
      setRouge(data.rouge)
    } catch (e) {
      setError(toErrorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card
      title="Evaluation (ROUGE)"
      subtitle="Word overlap with a human-written reference summary, 0–100 (higher means more overlap)"
    >
      {rouge ? (
        <>
          <RougeChart rouge={rouge} />
          <RougeTable rouge={rouge} />
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            ROUGE measures shared words and word order, not meaning: a correct paraphrase can score low, and it says
            nothing about factual accuracy.
          </p>
        </>
      ) : (
        <div className="space-y-3">
          <Banner kind="info" title="ROUGE not calculated">
            ROUGE compares a summary with a human-written reference summary. No reference was provided, so no score is
            shown. Paste one below to evaluate this summary.
          </Banner>
          <textarea
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            rows={3}
            placeholder="Paste a reference summary…"
            aria-label="Reference summary"
            className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          <Button variant="primary" onClick={evaluate} disabled={!reference.trim() || busy}>
            {busy ? 'Calculating…' : 'Calculate ROUGE'}
          </Button>
          {error && <Banner kind="error">{error}</Banner>}
        </div>
      )}
    </Card>
  )
}
