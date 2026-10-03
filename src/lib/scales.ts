/**
 * Chart scales shared by every figure.
 *
 * The compute axis is log10 effective compute C = K * N * 1536 * 4096 * (round+1),
 * the same quantity the paper's figures plot. Keeping it in one tested place is
 * what lets a point on screen be traced back to a paper point.
 */
import { scaleLinear, scaleLog } from 'd3-scale'
import type { AxisDomain } from './types'

export type { AxisDomain } from './types'

export interface Margin { top: number; right: number; bottom: number; left: number }

export const DEFAULT_MARGIN: Margin = { top: 16, right: 24, bottom: 44, left: 60 }

/** Loss axis domain: bits per byte. 8 is the uniform-over-256 reference. */
export function lossDomain(values: number[]): AxisDomain {
  const finite = values.filter((v) => Number.isFinite(v) && v > 0)
  if (finite.length === 0) return [0, 8]
  const lo = Math.min(...finite)
  const hi = Math.max(...finite)
  const pad = Math.max((hi - lo) * 0.06, 0.05)
  return [Math.max(0, lo - pad), hi + pad]
}

export function makeLogX(values: number[], range: [number, number]) {
  const finite = values.filter((v) => Number.isFinite(v) && v > 0)
  if (finite.length === 0) return scaleLog().domain([1, 10]).range(range)
  return scaleLog()
    .domain([Math.min(...finite), Math.max(...finite)])
    .range(range)
    .clamp(true)
}

export function makeLinearY(domain: AxisDomain, range: [number, number]) {
  return scaleLinear().domain(domain).range(range).clamp(true)
}

/** Evaluate the fitted law `L(C) = floor + amplitude * C^-alpha`. */
export function predictFit(
  fit: { amplitude: number; floor: number; alpha: number },
  compute: number,
): number {
  return fit.floor + fit.amplitude * Math.pow(compute, -fit.alpha)
}

/**
 * Chinchilla-form conversion from Table 2's caption:
 * a published (alpha, beta) data-scaling fit implies a compute exponent
 * b = alpha*beta/(alpha + beta) under compute-optimal allocation.
 */
export function chinchillaToComputeExponent(alpha: number, beta: number): number {
  return (alpha * beta) / (alpha + beta)
}

/** Format a compute value as a readable FLOP-equivalent count. */
export function formatCompute(c: number): string {
  if (!Number.isFinite(c) || c <= 0) return '—'
  const units: [number, string][] = [
    [1e18, 'E'], [1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'],
  ]
  for (const [scale, suffix] of units) {
    if (c >= scale) return `${(c / scale).toFixed(c / scale >= 100 ? 0 : 1)}${suffix}`
  }
  return c.toFixed(0)
}

/** Superscript-free exponent label for an axis tick, e.g. "10¹⁸". */
const SUPERSCRIPTS: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻',
}

export function formatPowerOfTen(exponent: number): string {
  const digits = String(Math.round(exponent))
    .split('')
    .map((d) => SUPERSCRIPTS[d] ?? d)
    .join('')
  return `10${digits}`
}

/** Log10 decade ticks covering a domain, e.g. [1e15, 1e16, ...]. */
export function decadeTicks(domain: [number, number]): number[] {
  const [lo, hi] = domain
  if (!(lo > 0) || !(hi > 0)) return []
  const ticks: number[] = []
  const start = Math.floor(Math.log10(lo))
  const end = Math.ceil(Math.log10(hi))
  // Guard against a pathological domain producing thousands of ticks.
  if (end - start > 24) return []
  for (let e = start; e <= end; e++) ticks.push(Math.pow(10, e))
  return ticks.filter((t) => t >= lo && t <= hi)
}