/**
 * Types mirroring public/data/*.json, emitted by tools/build_data.py.
 *
 * These are hand-maintained and checked against the bundle by
 * tools/verify_data.py -- if the generator's shape changes and this file does
 * not, the build is wrong and the mismatch shows up as a type error rather than
 * as `undefined` at runtime.
 */

// ---------------------------------------------------------------------------
// Paper references (D6: every claim carries one)
// ---------------------------------------------------------------------------

/** Key into `Meta.sections`. Unresolvable keys fail tools/verify_data.py. */
export type SectionKey =
  | 'abstract'
  | 'sec1' | 'sec2' | 'sec2.1' | 'sec2.2' | 'sec3.1' | 'sec3.2' | 'sec4'
  | 'sec5' | 'sec6'
  | 'appA.1' | 'appA.2' | 'appB' | 'appC' | 'appD' | 'appE' | 'appF'
  | 'appG' | 'appH'
  | 'fig1' | 'fig2' | 'fig3' | 'fig4' | 'fig5' | 'fig6' | 'fig7'
  | 'fig8' | 'fig9' | 'fig10'
  | 'eq1' | 'eq2' | 'eq3' | 'eq4' | 'eq5' | 'eq6' | 'eq7'
  | 'table1' | 'table2' | 'table3' | 'table4' | 'table5'

export type Sections = Record<SectionKey, string>

/** A claim as rendered in the UI: a statement plus where it comes from. */
export interface Sourced {
  ref: SectionKey
  /** Optional extra references, e.g. a table plus its appendix. */
  alsoRef?: SectionKey[]
}

// ---------------------------------------------------------------------------
// meta.json
// ---------------------------------------------------------------------------

export interface PaperInfo {
  title: string
  arxiv: string
  arxivVersion: string
  date: string
  authors: string[]
  url: string
  codeRepo: string
  weightsRepo: string
}

export interface Rung {
  rung: string
  label: string
  paper_params: number
  d_model: number
  n_heads: number
  n_layers: number
  released_params: number
}

export type ArmKey = 'selfplay' | 'uniform' | 'pcfg'

export interface ArmInfo {
  label: string
  short: string
  color: string
  /** SVG stroke-dasharray, or null for a solid line. */
  dash: string | null
  blurb: string
}

export interface CorpusInfo {
  key: string
  paperLabel: string
  group: string
  inTable2: boolean
}

export interface LiteratureEntry {
  /** Directly published compute exponents. */
  values: number[]
  /** Chinchilla-form (alpha, beta) pairs; converted via b=ab/(a+b) in the UI. */
  chinchilla: [number, number][]
  extraValues: number[]
  /** The paper prints a dash: no published exponent was found. */
  dash: boolean
  refs: string[]
}

export interface BfMacro { token: string; expansion: string; effect: string }
export interface BfPrimitive { token: string; effect: string }

export interface RewardArmInfo extends Sourced {
  key: string
  label: string
  included: boolean
  desc: string
}

export interface IclTaskInfo extends Sourced {
  label: string
  blurb: string
}

export interface Meta {
  paper: PaperInfo
  sections: Sections
  compute: { pool: number; context: number; formula: string }
  rungs: Rung[]
  arms: Record<ArmKey, ArmInfo>
  corpora: CorpusInfo[]
  literature: Record<string, LiteratureEntry>
  bf: {
    primitives: BfPrimitive[]
    macros: BfMacro[]
    params: {
      cell_modulus: number
      program_alphabet: string
      prefix_program: string
      prefix_output: string
    }
  }
  iclTasks: Record<string, IclTaskInfo>
  iclExtraTasks: Record<string, IclTaskInfo>
  rewardArms: RewardArmInfo[]
  mathPrior: {
    totalSamples: number
    programsPerRound: number
    ruleOfThree: number
  }
  provenance: {
    vendorRepo: string
    vendorCommit: string
    note: string
    license: string
  }
}

// ---------------------------------------------------------------------------
// scaling.json
// ---------------------------------------------------------------------------

/**
 * Columnar point table. Parallel arrays indexed by position; `ci` and `ri` index
 * into `stringTables`. Sorted by (corpus, compute).
 */
export interface Columnar {
  stringTables: { rungs: string[]; corpora: string[] }
  columns: {
    N: number[]
    round: number[]
    K: number[]
    /** Corpus index into stringTables.corpora. */
    ci: number[]
    bpb: number[]
    /** Effective compute C = K * N * POOL * CTX * (round + 1). */
    C: number[]
  }
}

/** Fitted `L(C) = floor + amplitude * C^-alpha`, from the authors' own code. */
export interface PowerFit {
  alpha: number
  amplitude: number
  floor: number
  rmse: number
  nPoints: number
}

export interface CorpusScaling {
  /** Ascending [compute, loss] pairs forming the compute-optimal frontier. */
  frontier: [number, number][]
  nFrontier: number
  bestLoss: number
  maxCompute: number
  /** null when the frontier is too short to fit (the paper's own guard). */
  fit: PowerFit | null
}

export interface ArmScaling extends Columnar {
  corpora: Record<string, CorpusScaling>
}

export interface ExponentRow {
  key: string
  paperLabel: string
  group: string
  alpha: number | null
  amplitude: number | null
  floor: number | null
  rmse: number | null
  nPoints: number
  nFrontier: number
  literatureValues: number[]
  literatureChinchilla: number[]
  literatureDash: boolean
  refs: string[]
}

export interface Scaling {
  /** Source of Table 2 and the Figure 1 scaling panel. Never mix with arms. */
  ladder: ArmScaling
  ladderProvenance: Record<string, unknown>
  /** Figure 2's three-arm comparison. */
  arms: Record<ArmKey, ArmScaling>
  exponents: ExponentRow[]
}

// ---------------------------------------------------------------------------
// table5.json
// ---------------------------------------------------------------------------

export interface Table5Cell { bpb: number; kUsed: number }

export interface Table5Column extends Sourced {
  key: string
  label: string
  desc: string
}

export interface Table5Row {
  key: string
  label: string
  /** One entry per column, in `columns` order. null where unscored. */
  cells: (Table5Cell | null)[]
  /** Index of the lowest-loss arm at the paper's printed precision. */
  bestIndex: number | null
}

export interface Table5 {
  rung: string
  round: number
  K: number
  columns: Table5Column[]
  rows: Table5Row[]
  provenance: Record<string, unknown> | null
  notes: string[]
}

// ---------------------------------------------------------------------------
// math.json
// ---------------------------------------------------------------------------

export type MathFamily =
  | 'arithmetic' | 'fibonacci' | 'geometric' | 'quadratic' | 'cubic'

export interface MathExample {
  program: string
  terms: number[]
  period: number
  params: Record<string, unknown>
  startOffset: number
  tapeLen: number
  rung: string
  round: number
}

export interface MathFamilyEntry extends Sourced {
  family: MathFamily
  label: string
  /** Earliest round in the paper's Table 1. */
  selfplay_round: number
  /** Expected first round under uniform sampling (Appendix C). */
  prior_expected_round: number
  prior_p: number | null
  prior_hits: number
  /** The paper's illustrative example program. */
  program: string
  terms: string
  hitCount: number
  /** Earliest round recomputed from hits.jsonl; must equal selfplay_round. */
  earliestRoundDerived: number | null
  rungsSeen: number
  seedsSeen: number
  example: MathExample | null
}

export interface MathProgram {
  family: MathFamily
  round: number
  rung: string
  seed: string
  program: string
  terms: number[]
  period: number
}

export interface MathData {
  families: MathFamilyEntry[]
  programs: MathProgram[]
  priorCounts: {
    description: string
    programs_per_round: number
    samples: number
    hit_counts: Record<string, number>
  }
  totalHits: number
  totalDistinctPrograms: number
}

// ---------------------------------------------------------------------------
// icl.json
// ---------------------------------------------------------------------------

export interface IclRow {
  task: string
  /** In-context examples; null for harness variants that do not sweep it. */
  m: number | null
  /** Arity or dictionary size, depending on the task. */
  k: number | null
  /** Metric name -> value. Names differ per harness version, so all are kept. */
  metrics: Record<string, number>
}

export interface IclFile {
  rows: IclRow[]
  ms: number[] | null
  ks: number[] | null
  format: string | null
  pushToken: number | null
  popToken: number | null
}

export interface IclArm {
  model: string | null
  trials: number | null
  files: Record<string, IclFile>
}

export interface SumBehavior {
  m: number
  correctAnswer: number
  low4BitsCorrect: number
  preferredBytes: number
  contextByte: number
  other: number
  entropyMeanBits: number
  entropyQ25Bits: number
  entropyQ75Bits: number
}

export interface IclData {
  arms: Record<string, IclArm>
  sumBehavior: SumBehavior[]
  categories: { key: keyof SumBehavior | string; label: string; color: string }[]
}

// ---------------------------------------------------------------------------
// curriculum.json, prepretraining.json, encodings.json
// ---------------------------------------------------------------------------

export interface CurriculumRow {
  arm: string
  T: number
  epiplexityMean: number
  epiplexitySd: number
  epiplexitySeeds: number[]
  dclmSeedMean: number
  dclmSubsetMean: number
  dclmSubsetSd: number
  dclmEnsembleK8: number
  audioSeedMean: number
  audioSubsetMean: number
  audioSubsetSd: number
  audioEnsembleK8: number
  cifarSeedMean: number
  cifarSubsetMean: number
  cifarSubsetSd: number
  cifarEnsembleK8: number
}

export interface CurriculumData { rows: CurriculumRow[] }

export interface PreTrainingArm {
  lr: string | null
  wd: string | null
  convergence: unknown
  curve: { tokens: number; bpb: number }[]
}

export interface PreTrainingData {
  modalities: Record<
    string,
    { scratch: PreTrainingArm; warm: PreTrainingArm }
  >
}

export interface EncodingsData {
  samples: Record<string, number[][]>
  contextLength: number
}

// ---------------------------------------------------------------------------
// local_measurements.json (D5: our own CPU scoring, not the paper's)
// ---------------------------------------------------------------------------

export interface LocalMeasurement {
  round: number
  bitsPerByte: number
  top1Accuracy: number
  isRandomInit: boolean
}

export interface LocalMeasurements {
  points: LocalMeasurement[]
  provenance: {
    what: string
    weightsRepo: string
    rung: string
    seed: number
    params: number
    corpus: string
    nSequences: number
    contextLength: number
    uniformBaseline: number
    host: {
      cpu: string
      torchThreads: number
      torch: string
      python: string
      gpu: string
    }
  }
  interpretation: { control: string; claim: string; caveat: string; reference: string }
}