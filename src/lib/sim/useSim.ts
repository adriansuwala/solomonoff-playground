/**
 * React glue for the simulation worker.
 *
 * The contract this hook maintains, and why each part is here:
 *
 *   - The page must never block. A 200-round run is ~2.8 s of arithmetic; run
 *     inline it freezes the tab long enough that the browser stops answering
 *     script evaluation, which is exactly what happened before this existed.
 *   - A stale result must never overwrite a fresh one. Switching curriculum
 *     mid-run leaves an in-flight request behind, so every result is matched
 *     against the request id that asked for it and discarded if it lost the
 *     race.
 *   - Unmount must not leak a worker or a pending timer.
 *   - Failure must be visible. A silently blank exhibit is indistinguishable
 *     from a sim that found nothing, which is the failure mode this page is
 *     most vulnerable to.
 */
import { useEffect, useRef, useState } from 'react'
import type { SimResult } from './runsim'
import type { TeachKind, WorkerRequest, WorkerResponse } from './simWorker'

export interface SimState {
  fixed: SimResult | null
  moving: SimResult | null
  /** True while either arm is still running. */
  busy: boolean
  error: string | null
}

export interface SimOptions {
  rounds: number
  recordEvery: number
  span: number
  /** Skip the worker entirely. Tests use this; the page never does. */
  disabled?: boolean
}

const EMPTY: SimState = { fixed: null, moving: null, busy: true, error: null }

export function useSimRuns(opts: SimOptions): SimState {
  const [state, setState] = useState<SimState>(EMPTY)
  const seq = useRef(0)

  const { rounds, recordEvery, span } = opts

  useEffect(() => {
    // One worker for both arms. Two would double the memory for a computation
    // that is embarrassingly parallel but only ~3 s each.
    let worker: Worker | null = null
    try {
      worker = new Worker(new URL('./simWorker.ts', import.meta.url), { type: 'module' })
    } catch (err) {
      setState({
        fixed: null,
        moving: null,
        busy: false,
        error: `could not start the simulation worker: ${
          err instanceof Error ? err.message : String(err)
        }`,
      })
      return
    }

    // Keyed by arm name, because that is what the state is indexed by.
    const pending = new Map<string, 'fixed' | 'moving'>()
    const mine = ++seq.current

    worker.onmessage = (ev: MessageEvent<WorkerResponse>): void => {
      const msg = ev.data
      const which = pending.get(msg.id)
      // Drop anything from a superseded request, or from an unmounted hook.
      if (which === undefined || mine !== seq.current) return
      pending.delete(msg.id)

      // Narrow on the discriminant before touching arm-specific fields; reading
      // `.error` off the union would not typecheck and `.result` is meaningless
      // on the failure arm.
      if (!msg.ok) {
        setState((s) => ({ ...s, busy: pending.size > 0, error: msg.error }))
        return
      }
      setState((s) => ({ ...s, [which]: msg.result, busy: pending.size > 0 }))
    }

    worker.onerror = (ev: ErrorEvent): void => {
      setState({
        fixed: null,
        moving: null,
        busy: false,
        error: `simulation worker failed: ${ev.message || 'unknown error'}`,
      })
    }

    // `moving` is this hook's name for an arm; the worker calls the same thing
    // `stepped`. Mapping explicitly keeps the two vocabularies from drifting.
    const arms: { arm: 'fixed' | 'moving'; teach: TeachKind }[] = [
      { arm: 'fixed', teach: 'fixed' },
      { arm: 'moving', teach: 'stepped' },
    ]
    for (const { arm, teach } of arms) {
      const req: WorkerRequest = {
        id: `${arm}-${rounds}-${recordEvery}-${span}`,
        rounds,
        recordEvery,
        teach,
        span,
      }
      pending.set(req.id, arm)
      worker.postMessage(req)
    }

    return () => {
      seq.current++
      worker.terminate()
    }
  }, [rounds, recordEvery, span])

  return state
}