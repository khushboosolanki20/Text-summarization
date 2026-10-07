import { useEffect, useState } from 'react'
import { getHealth, toErrorMessage } from '../services/api'

const STYLES = {
  checking: 'bg-slate-100 text-slate-600 ring-slate-200',
  online: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  offline: 'bg-rose-50 text-rose-700 ring-rose-200',
}

const DOT = {
  checking: 'bg-slate-400 animate-pulse',
  online: 'bg-emerald-500',
  offline: 'bg-rose-500',
}

/** Small pill that shows whether the FastAPI backend is reachable. */
export default function BackendStatus() {
  const [state, setState] = useState({ status: 'checking', info: null, error: null })

  useEffect(() => {
    let cancelled = false
    getHealth()
      .then((info) => !cancelled && setState({ status: 'online', info, error: null }))
      .catch((err) => !cancelled && setState({ status: 'offline', info: null, error: toErrorMessage(err) }))
    return () => {
      cancelled = true
    }
  }, [])

  const { status, info, error } = state
  const label =
    status === 'checking'
      ? 'Connecting to backend…'
      : status === 'online'
        ? `Backend online · v${info.version}`
        : 'Backend offline'

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ring-1 ${STYLES[status]}`}
      title={error ?? undefined}
    >
      <span className={`h-2 w-2 rounded-full ${DOT[status]}`} />
      {label}
    </div>
  )
}
