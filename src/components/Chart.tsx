/**
 * Chart primitives: axes, grid, lines, points.
 *
 * Hand-rolled SVG rather than a charting library (D4) so the compute axis
 * definition stays visible and each point stays traceable to a paper point.
 */
import { scaleLinear } from 'd3-scale'
import { useMemo, useState, type ReactNode } from 'react'
import {
  decadeTicks, formatCompute, formatPowerOfTen, makeLinearY, makeLogX,
  type Margin,
} from '@/lib/scales'
import type { AxisDomain } from '@/lib/types'

export interface ChartFrameProps {
  width?: number
  height?: number
  margin?: Partial<Margin>
  xDomain: AxisDomain
  yDomain: AxisDomain
  xLabel: string
  yLabel: string
  /**
   * 'log' is the default and correct for effective compute. Use 'linear' only
   * for genuinely linear axes such as round index -- a log scale cannot
   * represent 0, and forcing one produces an undefined tick.
   */
  xScale?: 'log' | 'linear'
  /** Suppress the power-of-ten tick row when the axis is not log. */
  children: (geom: {
    x: (c: number) => number
    y: (l: number) => number
    innerWidth: number
    innerHeight: number
    plot: DOMRect | null
  }) => ReactNode
  onHover?: (payload: TooltipPayload | null) => void
}

/** Tooltip content, positioned by the caller. */
export interface TooltipPayload {
  x: number
  y: number
  title: string
  rows: [string, string][]
}

const M = { top: 16, right: 24, bottom: 44, left: 60 }

export function ChartFrame({
  width = 720, height = 380, margin, xDomain, yDomain, xLabel, yLabel,
  xScale = 'log', children,
}: ChartFrameProps) {
  const m = { ...M, ...margin }
  const innerWidth = Math.max(10, width - m.left - m.right)
  const innerHeight = Math.max(10, height - m.top - m.bottom)
  const x = useMemo(
    () => (xScale === 'log'
      ? makeLogX(xDomain, [0, innerWidth])
      : scaleLinear().domain(xDomain).range([0, innerWidth]).clamp(true)),
    [xScale, xDomain, innerWidth],
  )
  const y = useMemo(() => makeLinearY(yDomain, [innerHeight, 0]), [yDomain, innerHeight])
  const xticks = useMemo(
    () => (xScale === 'log' ? decadeTicks(xDomain) : niceTicks(xDomain, 6)),
    [xScale, xDomain],
  )
  const yticks = useMemo(() => niceTicks(yDomain, 6), [yDomain])

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${yLabel} against ${xLabel}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <g transform={`translate(${m.left},${m.top})`}>
        <g className="chart__grid">
          {yticks.map((t) => (
            <line key={`gy${t}`} x1={0} x2={innerWidth} y1={y(t)} y2={y(t)} />
          ))}
        </g>
        <g className="chart__tick">
          {yticks.map((t) => (
            <text key={`ty${t}`} x={-8} y={y(t)} dy="0.32em" textAnchor="end">
              {formatTick(t)}
            </text>
          ))}
        </g>
        <g className="chart__tick">
          {xticks.map((t) => (
            <text
              key={`tx${t}`}
              x={x(t)} y={innerHeight + 18}
              textAnchor="middle"
            >
              {xScale === 'log'
                ? formatPowerOfTen(Math.round(Math.log10(t)))
                : formatTick(t)}
            </text>
          ))}
        </g>
        <line
          x1={0} x2={innerWidth} y1={innerHeight} y2={innerHeight}
          stroke="var(--border-strong)"
        />
        {children({ x: (c) => x(c), y: (l) => y(l), innerWidth, innerHeight, plot: null })}
        <text
          className="chart__axis-label"
          x={innerWidth / 2} y={innerHeight + 38} textAnchor="middle"
        >
          {xLabel}
        </text>
        <text
          className="chart__axis-label"
          transform={`translate(${-m.left + 12},${innerHeight / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yLabel}
        </text>
      </g>
    </svg>
  )
}

/** Evenly spaced round ticks covering a domain. */
export function niceTicks([lo, hi]: AxisDomain, count: number): number[] {
  if (!(hi > lo)) return [lo]
  const span = hi - lo
  const rawStep = span / count
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const norm = rawStep / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const out: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Number(t.toFixed(10)))
  }
  return out
}

export function formatTick(v: number): string {
  if (v === 0) return '0'
  if (Math.abs(v) >= 1000) return v.toExponential(0)
  if (Math.abs(v) >= 10) return v.toFixed(0)
  if (Math.abs(v) >= 1) return v.toFixed(1)
  return v.toFixed(2)
}

// ---------------------------------------------------------------------------
// Series
// ---------------------------------------------------------------------------

export interface SeriesProps {
  /** [compute, loss] pairs, ascending in compute. */
  points: [number, number][]
  color: string
  dash?: string | null
  width?: number
  opacity?: number
  showPoints?: boolean
  pointRadius?: number
}

export function Series({
  points, color, dash = null, width = 1.75, opacity = 1,
  showPoints = false, pointRadius = 2.5,
}: SeriesProps) {
  const d = useMemo(
    () => points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' '),
    [points],
  )
  return (
    <g>
      {points.length > 1 && (
        <path
          d={d} fill="none" stroke={color} strokeWidth={width}
          strokeDasharray={dash ?? undefined} opacity={opacity}
          strokeLinejoin="round"
        />
      )}
      {showPoints && points.map((p, i) => (
        <circle key={i} className="chart__point" cx={p[0]} cy={p[1]} r={pointRadius} fill={color} />
      ))}
    </g>
  )
}

// ---------------------------------------------------------------------------
// Hover readout
// ---------------------------------------------------------------------------

export interface HoverReadoutProps {
  payload: TooltipPayload | null
}

export function HoverReadout({ payload }: HoverReadoutProps) {
  if (!payload) return null
  return (
    <div
      className="tooltip"
      style={{ left: payload.x + 14, top: payload.y + 14 }}
      role="status"
    >
      <div style={{ marginBottom: 4, color: 'var(--text)' }}>{payload.title}</div>
      {payload.rows.map(([k, v]) => (
        <div className="tooltip__row" key={k}>
          <span style={{ color: 'var(--text-faint)' }}>{k}</span>
          <span>{v}</span>
        </div>
      ))}
    </div>
  )
}

/** Wrapper that tracks the pointer for a tooltip. */
export function useTooltip() {
  const [payload, setPayload] = useState<TooltipPayload | null>(null)
  return { payload, setPayload }
}

export function computeLabel(c: number): string {
  return formatCompute(c)
}