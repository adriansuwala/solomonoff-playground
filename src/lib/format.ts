/**
 * Number formatting shared by the page modules.
 *
 * The null case matters: a value that was not measured must render as an em
 * dash, never as "0.0%" (which would read as a measurement) and never as
 * "NaN%". Every caller in the app passes a possibly-null value straight from
 * the bundle, so this is the single place that decision is made.
 */

/** A fraction as a percentage, e.g. 0.844 -> "84.4%". Nullish becomes an em dash. */
export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`