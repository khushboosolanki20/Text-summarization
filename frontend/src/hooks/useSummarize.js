import { useCallback, useEffect, useRef, useState } from 'react'
import { startFileJob, startTextJob, toErrorMessage, waitForJob } from '../services/api'
import { addToHistory } from '../utils/history'

const IDLE = { status: 'idle', progress: null, result: null, error: null, startedAt: null, sourceLabel: null, method: null }

/**
 * Runs a summarization as a background job and tracks its lifecycle:
 * idle -> running (with progress) -> done | error.
 * Results are added to the local history.
 */
export function useSummarize() {
  const [state, setState] = useState(IDLE)
  const abortRef = useRef(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  const run = useCallback(async ({ text, file, method, length, referenceSummary }) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const sourceLabel = file ? file.name : 'Pasted text'
    setState({ ...IDLE, status: 'running', startedAt: Date.now(), sourceLabel, method })

    try {
      const job = file
        ? await startFileJob({ file, method, length, referenceSummary })
        : await startTextJob({ text, method, length, referenceSummary })
      const result = await waitForJob(job.job_id, {
        signal: controller.signal,
        onProgress: (progress) => setState((s) => ({ ...s, progress })),
      })
      addToHistory(result, sourceLabel)
      setState((s) => ({ ...s, status: 'done', result, progress: null }))
    } catch (error) {
      if (error.cancelled || controller.signal.aborted) return
      setState((s) => ({ ...s, status: 'error', error: toErrorMessage(error), progress: null }))
    }
  }, [])

  /** Stop waiting for the current job (the server finishes it in the background). */
  const cancel = useCallback(() => {
    abortRef.current?.abort()
    setState(IDLE)
  }, [])

  const reset = useCallback(() => setState(IDLE), [])

  return { ...state, run, cancel, reset }
}
