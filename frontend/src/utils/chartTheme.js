/**
 * Chart colors and chrome. Values come from a validated palette
 * (colorblind-safety and contrast checked with a palette validator):
 *
 * - SERIES: categorical slots in fixed order (never cycled). Slots 1-3 pass
 *   all-pairs CVD separation; slot 3 (aqua) is below 3:1 contrast, so every
 *   chart using it also ships a table view with the values.
 * - EMPHASIS: two steps of one blue ramp for "selected vs. other" (an ordinal
 *   pair, validated: monotone lightness, light end >= 2:1 vs. surface).
 * - Text never uses series colors; it uses the INK tokens.
 */
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a']

export const EMPHASIS = { strong: '#256abf', soft: '#86b6ef' }

export const INK = {
  primary: '#0b0b0b',
  secondary: '#52514e',
  muted: '#898781',
  grid: '#e1e0d9',
  axis: '#c3c2b7',
  surface: '#fcfcfb',
}

/** Shared props for recessive axes. */
export const axisProps = {
  stroke: INK.axis,
  tick: { fill: INK.muted, fontSize: 12 },
  tickLine: false,
}

export const gridProps = { stroke: INK.grid, vertical: false }

/** Bars: thin, rounded only at the data end. */
export const BAR_MAX_SIZE = 24
export const roundedTop = [4, 4, 0, 0]
export const roundedRight = [0, 4, 4, 0]
