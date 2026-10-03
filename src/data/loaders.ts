/**
 * Typed loaders for public/data/*.json.
 *
 * Everything is fetched once and cached. The bundle is ~10 MB, so pages pull
 * only the files they need: the scaling explorer never downloads math.json.
 */
import type {
  CurriculumData, EncodingsData, IclData, LocalMeasurements, MathData,
  Meta, PreTrainingData, Scaling, Table5,
} from './types'

const BASE = `${import.meta.env.BASE_URL}data`

const cache = new Map<string, Promise<unknown>>()

function fetchOnce<T>(file: string): Promise<T> {
  const hit = cache.get(file)
  if (hit) return hit as Promise<T>
  const p = fetch(`${BASE}/${file}`).then((r) => {
    if (!r.ok) throw new Error(`failed to load ${file}: ${r.status} ${r.statusText}`)
    return r.json() as Promise<T>
  })
  cache.set(file, p)
  return p
}

export const loadMeta = () => fetchOnce<Meta>('meta.json')
export const loadScaling = () => fetchOnce<Scaling>('scaling.json')
export const loadTable5 = () => fetchOnce<Table5>('table5.json')
export const loadMath = () => fetchOnce<MathData>('math.json')
export const loadIcl = () => fetchOnce<IclData>('icl.json')
export const loadCurriculum = () => fetchOnce<CurriculumData>('curriculum.json')
export const loadPreTraining = () => fetchOnce<PreTrainingData>('prepretraining.json')
export const loadEncodings = () => fetchOnce<EncodingsData>('encodings.json')
export const loadLocalMeasurements = () =>
  fetchOnce<LocalMeasurements>('local_measurements.json')