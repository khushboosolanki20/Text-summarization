import { useEffect, useState } from 'react'
import { formatSeconds } from '../utils/format'
import { Button, Card } from './ui/primitives'

/** Live job status: current stage, a determinate bar when the stage has steps, and elapsed time. */
export default function ProgressPanel({ progress, startedAt, method, onCancel }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(timer)
  }, [])

  const stage = progress?.stage ?? 'Starting…'
  const hasSteps = progress && progress.total > 1
  const percent = hasSteps ? Math.round((progress.done / progress.total) * 100) : null
  const abstractive = method === 'bart' || method === 'hybrid'

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="min-w-0 flex-1" aria-live="polite">
          <p className="text-sm font-medium text-slate-500">Working on it · {formatSeconds((now - startedAt) / 1000)}</p>
          <p className="mt-1 truncate text-lg font-semibold capitalize text-slate-900">{stage}</p>
          {hasSteps && (
            <p className="text-sm text-slate-600">
              {progress.done} of {progress.total}
            </p>
          )}
        </div>
        <Button variant="ghost" icon="x" onClick={onCancel}>
          Cancel
        </Button>
      </div>

      <div
        className="mt-4 h-2 overflow-hidden rounded-full bg-indigo-100"
        role="progressbar"
        aria-label="Summarization progress"
        aria-valuenow={percent ?? undefined}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        {percent != null ? (
          <div className="h-full rounded-full bg-indigo-600 transition-all duration-500" style={{ width: `${percent}%` }} />
        ) : (
          <div className="h-full w-1/3 animate-pulse rounded-full bg-indigo-500" />
        )}
      </div>

      {abstractive && (
        <p className="mt-3 text-xs text-slate-500">
          BART runs locally. The first request loads the model, and long documents are summarized chunk by chunk, which
          can take a few minutes on a CPU.
        </p>
      )}
    </Card>
  )
}
