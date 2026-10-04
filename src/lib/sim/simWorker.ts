/**
 * The simulation, run off the main thread.
 *
 * Why a worker and not just `useMemo`: a 200-round run scores the whole pool at
 * every recorded round, and measured on this box that is ~2.8 s of solid
 * arithmetic per run. The page needs two of them (a fixed and a moving
 * curriculum), so doing it inline blocks the main thread for ~5.5 s and the tab
 * is dead -- long enough that the browser will not even answer a script
 * evaluation while it happens.
 *
 * This is NOT the incremental live-round loop of Phase 4. It is the same pure
 * `runSim`, unchanged, moved to another thread. Phase 4 makes the loop
 * steppable; this just stops it freezing the page, and the two compose because
 * the worker calls the identical function the tests pin.
 */
import { runSim, type RunConfig } from './runsim'
import { poolFor } from './programs'
import type { SimResult } from './runsim'

/** One curriculum arm's worth of work. */
export interface SimRequest {
  id: string
  cfg: RunConfig
}

/**
 * runSim is a plain function, so it cannot cross the worker boundary. The
 * curriculum selector is the one piece of behaviour that has to, and it is
 * passed by name rather than as a closure.
 */
export type TeachKind = 'fixed' | 'stepped'

export interface WorkerRequest {
  id: string
  rounds: number
  recordEvery: number
  teach: TeachKind
  span: number
}

export type WorkerResponse =
  | { id: string; ok: true; result: SimResult }
  | { id: string; ok: false; error: string }

// Imported lazily inside the handler so a syntax error in the library surfaces
// as a rejected promise the page can display, rather than a worker that dies
// silently at import time with nothing on screen.
self.onmessage = (ev: MessageEvent<WorkerRequest>): void => {
  const { id, rounds, recordEvery, teach, span } = ev.data
  void (async () => {
    try {
      const { fixedTeach, steppedTeach, DEFAULT_RUN } = await import('./runsim')
      const pool = poolFor()
      const result = runSim(pool, {
        ...DEFAULT_RUN,
        rounds,
        recordEvery,
        poolSize: pool.length,
        teach: teach === 'fixed' ? fixedTeach : steppedTeach(span),
      })
      const msg: WorkerResponse = { id, ok: true, result }
      self.postMessage(msg)
    } catch (err) {
      const msg: WorkerResponse = {
        id,
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      }
      self.postMessage(msg)
    }
  })()
}