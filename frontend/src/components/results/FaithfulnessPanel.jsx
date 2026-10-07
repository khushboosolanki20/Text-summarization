import Icon from '../ui/Icon'
import { Badge, Card } from '../ui/primitives'

/**
 * Experimental "potentially unsupported content" check for BART / Hybrid.
 * Each flagged sentence shows why it was flagged and the closest source
 * passage, so the reader can verify it. Labelled clearly as a heuristic.
 */
export default function FaithfulnessPanel({ result }) {
  const report = result.faithfulness
  if (!report?.applicable) return null

  const flagged = report.sentences.filter((s) => s.flagged)
  const passage = (indices) => indices.map((i) => result.sentences[i]?.text).filter(Boolean).join(' ')

  return (
    <Card
      title={
        <span className="flex flex-wrap items-center gap-2">
          Potentially unsupported content <Badge tone="amber">Experimental</Badge>
        </span>
      }
      subtitle={report.note}
    >
      {flagged.length === 0 ? (
        <div className="flex items-start gap-3 rounded-xl bg-emerald-50 p-4 text-sm text-emerald-900 ring-1 ring-emerald-200">
          <Icon name="check" className="mt-0.5 h-5 w-5 shrink-0" />
          <p>
            <span className="font-semibold">No sentences flagged.</span> All {report.sentences.length} summary sentences
            use words, numbers and names found in the source. This does not prove the summary is correct.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-sm text-slate-600">
            {flagged.length} of {report.sentences.length} summary sentences may contain information that is not in the
            source. Please check them against the original.
          </p>
          {flagged.map((s, i) => (
            <div key={i} className="rounded-xl border border-amber-200 bg-amber-50/60 p-4">
              <p className="flex items-start gap-2 font-medium text-slate-900">
                <Icon name="alert" className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
                <span>
                  <span className="sr-only">Warning: </span>
                  {s.sentence}
                </span>
              </p>
              <ul className="mt-2 list-disc space-y-0.5 pl-12 text-sm text-amber-900">
                {s.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
              {s.closest_source.length > 0 && (
                <div className="mt-3 rounded-lg bg-white p-3 text-sm ring-1 ring-slate-200">
                  <p className="text-xs font-medium text-slate-500">
                    Closest source passage (similarity {s.similarity.toFixed(2)})
                  </p>
                  <p className="mt-1 text-slate-700">{passage(s.closest_source)}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
