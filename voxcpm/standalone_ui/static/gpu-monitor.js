import { browserLanguage, t } from './i18n.js'

const SESSION_KEY = 'voxcpmtts-gpu-history-v1'
const HISTORY_RETENTION_MS = 10 * 60 * 1000
const POLL_INTERVAL_MS = 1000
const WINDOWS = [60 * 1000, 10 * 60 * 1000]

const METRICS = [
  { key: 'utilization', label: () => t('gpu.metric.gpu', {}, 'GPU utilization'), color: '#ff7a1a' },
  { key: 'memory_utilization', label: () => t('gpu.metric.memoryActivity', {}, 'Memory activity'), color: '#c586c0' },
  { key: 'memory_used', label: () => t('gpu.metric.vram', {}, 'VRAM'), color: '#72a7ff' },
  { key: 'temperature', label: () => t('gpu.metric.temperature', {}, 'Temperature'), color: '#ef6b73' },
  { key: 'power', label: () => t('gpu.metric.power', {}, 'Power'), color: '#f2c94c' },
  { key: 'fan_speed', label: () => t('gpu.metric.fan', {}, 'Fan'), color: '#55c58a' },
  { key: 'graphics_clock', label: () => t('gpu.metric.graphicsClock', {}, 'Graphics clock'), color: '#9cdcfe' },
  { key: 'memory_clock', label: () => t('gpu.metric.memoryClock', {}, 'Memory clock'), color: '#ce9178' },
]

function element(tag, className, text) {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function readSession() {
  try { return JSON.parse(sessionStorage.getItem(SESSION_KEY) || 'null') } catch { return null }
}

function historyPoints(samples, now, windowMs, metricKey, maximum, width = 300, height = 70) {
  const windowStart = now - windowMs
  return samples.filter((sample) => Number.isFinite(sample[metricKey])).map((sample) => {
    const x = Math.min(width, Math.max(0, (sample.timestamp - windowStart) / windowMs * width))
    const y = height - (Math.min(maximum, Math.max(0, sample[metricKey])) / maximum * height)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function addChartGrid(svg, width, height) {
  for (let column = 0; column <= 10; column += 1) {
    const x = column * width / 10
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.setAttribute('class', 'gpu-grid-line')
    line.setAttribute('x1', String(x))
    line.setAttribute('x2', String(x))
    line.setAttribute('y1', '0')
    line.setAttribute('y2', String(height))
    svg.append(line)
  }
  for (let row = 0; row <= 4; row += 1) {
    const y = row * height / 4
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.setAttribute('class', 'gpu-grid-line')
    line.setAttribute('x1', '0')
    line.setAttribute('x2', String(width))
    line.setAttribute('y1', String(y))
    line.setAttribute('y2', String(y))
    svg.append(line)
  }
}

function metricMaximum(metric, gpu, history) {
  const observed = Math.max(1, ...history.map((sample) => sample[metric.key] || 0))
  if (['utilization', 'memory_utilization', 'temperature', 'fan_speed'].includes(metric.key)) return 100
  if (metric.key === 'memory_used' && Number.isFinite(gpu.memory_total)) return Math.max(1, gpu.memory_total)
  if (metric.key === 'power' && Number.isFinite(gpu.power_limit)) return Math.max(1, gpu.power_limit)
  if (metric.key === 'graphics_clock' && Number.isFinite(gpu.graphics_clock_max)) return Math.max(1, gpu.graphics_clock_max)
  if (metric.key === 'memory_clock' && Number.isFinite(gpu.memory_clock_max)) return Math.max(1, gpu.memory_clock_max)
  return Math.ceil(observed * 1.1)
}

function formatMetric(metric, value) {
  if (!Number.isFinite(value)) return 'N/A'
  if (['utilization', 'memory_utilization', 'fan_speed'].includes(metric.key)) return `${Math.round(value)}%`
  if (metric.key === 'memory_used') return `${(value / 1024).toFixed(1)} GB`
  if (metric.key === 'temperature') return `${Math.round(value)} C`
  if (metric.key === 'power') return `${Math.round(value)} W`
  return `${Math.round(value)} MHz`
}

export class GpuMonitor {
  constructor(output, { fetchJson } = {}) {
    this.output = output
    this.fetchJson = fetchJson || this.defaultFetchJson
    this.history = new Map()
    this.stats = []
    this.windowMs = 60 * 1000
    this.timer = null
    this.refreshActive = false
    this.hovering = false
    this.restoreSession()
  }

  async defaultFetchJson(path) {
    const response = await fetch(path)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return response.json()
  }

  restoreSession() {
    const cached = readSession()
    const cutoff = Date.now() - HISTORY_RETENTION_MS
    if (!cached || !Number.isFinite(cached.savedAt) || cached.savedAt < cutoff) return
    if (WINDOWS.includes(cached.windowMs)) this.windowMs = cached.windowMs
    if (Array.isArray(cached.stats)) this.stats = cached.stats
    Object.entries(cached.history || {}).forEach(([index, samples]) => {
      const recent = Array.isArray(samples)
        ? samples.filter((sample) => Number.isFinite(sample?.timestamp) && sample.timestamp >= cutoff)
        : []
      if (recent.length) this.history.set(Number(index), recent)
    })
  }

  persistSession() {
    try {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({
        savedAt: Date.now(),
        windowMs: this.windowMs,
        stats: this.stats,
        history: Object.fromEntries(this.history),
      }))
    } catch {
      // Monitoring continues in memory when session storage is unavailable.
    }
  }

  mergeHistory(historyPayload) {
    const cutoff = Date.now() - HISTORY_RETENTION_MS
    Object.entries(historyPayload || {}).forEach(([index, incoming]) => {
      if (!Array.isArray(incoming)) return
      const byTimestamp = new Map()
      ;[...(this.history.get(Number(index)) || []), ...incoming].forEach((sample) => {
        if (Number.isFinite(sample?.timestamp) && sample.timestamp >= cutoff) byTimestamp.set(sample.timestamp, sample)
      })
      const merged = [...byTimestamp.values()].sort((left, right) => left.timestamp - right.timestamp)
      if (merged.length) this.history.set(Number(index), merged)
    })
    this.history.forEach((samples, index) => {
      const recent = samples.filter((sample) => sample.timestamp >= cutoff)
      if (recent.length) this.history.set(index, recent)
      else this.history.delete(index)
    })
    this.persistSession()
  }

  attachChartHover(plot, samples, metric, now) {
    const line = element('div', 'gpu-hover-line')
    const tooltip = element('div', 'gpu-hover-tooltip')
    line.hidden = true
    tooltip.hidden = true
    plot.append(line, tooltip)
    plot.addEventListener('pointermove', (event) => {
      this.hovering = true
      const bounds = plot.getBoundingClientRect()
      const offset = Math.min(bounds.width, Math.max(0, event.clientX - bounds.left))
      const ratio = bounds.width ? offset / bounds.width : 0
      const targetTime = now - this.windowMs + (ratio * this.windowMs)
      const nearest = samples.reduce((best, sample) => {
        if (!best) return sample
        return Math.abs(sample.timestamp - targetTime) < Math.abs(best.timestamp - targetTime) ? sample : best
      }, null)
      const tolerance = Math.max(1500, this.windowMs * 10 / Math.max(1, bounds.width))
      const hasSample = nearest && Math.abs(nearest.timestamp - targetTime) <= tolerance
      const shownTime = new Date(hasSample ? nearest.timestamp : targetTime).toLocaleTimeString(browserLanguage())
      tooltip.textContent = hasSample
        ? `${formatMetric(metric, nearest[metric.key])} / ${shownTime}`
        : t('gpu.noSampleAt', { time: shownTime }, `No sample / ${shownTime}`)
      const percent = ratio * 100
      line.style.left = `${percent}%`
      tooltip.style.left = `${percent}%`
      tooltip.classList.toggle('align-start', percent < 18)
      tooltip.classList.toggle('align-end', percent > 82)
      line.hidden = false
      tooltip.hidden = false
    })
    plot.addEventListener('pointerleave', () => {
      this.hovering = false
      line.hidden = true
      tooltip.hidden = true
      this.render(this.stats)
    })
  }

  createMetricChart(metric, gpu, history, now) {
    const current = gpu[metric.key]
    if (!Number.isFinite(current)) return null
    const label = metric.label()
    const samples = history.filter((sample) => Number.isFinite(sample[metric.key]))
    const maximum = metricMaximum(metric, gpu, samples)
    const average = samples.length ? samples.reduce((total, sample) => total + sample[metric.key], 0) / samples.length : current
    const peak = samples.length ? Math.max(...samples.map((sample) => sample[metric.key])) : current
    const chart = element('div', 'gpu-metric-chart')
    chart.style.setProperty('--chart-color', metric.color)
    const chartScale = element('div', 'gpu-chart-scale')
    chartScale.append(element('span', '', label), element('strong', '', formatMetric(metric, current)))
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('class', 'gpu-sparkline')
    svg.setAttribute('viewBox', '0 0 300 70')
    svg.setAttribute('preserveAspectRatio', 'none')
    svg.setAttribute('role', 'img')
    svg.setAttribute('aria-label', t('gpu.historyLabel', {
      metric: label,
      average: formatMetric(metric, average),
      peak: formatMetric(metric, peak),
    }, `${label} history, average ${formatMetric(metric, average)}, peak ${formatMetric(metric, peak)}`))
    addChartGrid(svg, 300, 70)
    const points = historyPoints(samples, now, this.windowMs, metric.key, maximum)
    if (samples.length > 1) {
      const pointList = points.split(' ')
      const area = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
      area.setAttribute('class', 'gpu-chart-area')
      area.setAttribute('points', `${pointList[0].split(',')[0]},70 ${points} ${pointList.at(-1).split(',')[0]},70`)
      svg.append(area)
    }
    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline')
    polyline.setAttribute('class', 'gpu-chart-line')
    polyline.setAttribute('points', points)
    svg.append(polyline)
    if (samples.length) {
      const latest = points.split(' ').at(-1).split(',')
      const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
      marker.setAttribute('class', 'gpu-chart-marker')
      marker.setAttribute('cx', latest[0])
      marker.setAttribute('cy', latest[1])
      marker.setAttribute('r', '2.5')
      svg.append(marker)
    }
    const plot = element('div', 'gpu-chart-plot')
    plot.append(svg)
    this.attachChartHover(plot, samples, metric, now)
    const axis = element('div', 'gpu-chart-axis')
    axis.append(
      element('span', '', this.windowMs === 60 * 1000 ? t('gpu.oneMinute', {}, '1 minute') : t('gpu.tenMinutes', {}, '10 minutes')),
      element('span', '', t('gpu.averagePeak', {
        average: formatMetric(metric, average),
        peak: formatMetric(metric, peak),
      }, `Average ${formatMetric(metric, average)} · Peak ${formatMetric(metric, peak)}`)),
    )
    chart.append(chartScale, plot, axis)
    return chart
  }

  render(gpus) {
    const monitor = element('div', 'gpu-monitor')
    const heading = element('div', 'gpu-monitor-heading')
    const title = element('div', 'gpu-monitor-title')
    title.append(element('i', 'icon-activity'), element('strong', '', t('gpu.telemetry', {}, 'GPU telemetry')))
    const windowControl = element('div', 'gpu-window-control')
    windowControl.setAttribute('role', 'group')
    windowControl.setAttribute('aria-label', t('gpu.historyWindow', {}, 'GPU history window'))
    ;[[60 * 1000, t('gpu.oneMinuteShort', {}, '1 min')], [10 * 60 * 1000, t('gpu.tenMinutesShort', {}, '10 min')]].forEach(([windowMs, label]) => {
      const button = element('button', windowMs === this.windowMs ? 'active' : '', label)
      button.type = 'button'
      button.setAttribute('aria-pressed', String(windowMs === this.windowMs))
      button.addEventListener('click', () => {
        this.windowMs = windowMs
        this.persistSession()
        this.render(this.stats)
      })
      windowControl.append(button)
    })
    heading.append(title, windowControl)
    monitor.append(heading)
    if (!gpus.length) {
      monitor.append(element('div', 'gpu-monitor-muted', t('gpu.unavailable', {}, 'GPU telemetry unavailable.')))
      this.output.replaceChildren(monitor)
      return
    }

    const grid = element('div', 'gpu-card-grid')
    gpus.forEach((gpu) => {
      const now = Date.now()
      const history = (this.history.get(gpu.index) || []).filter((sample) => sample.timestamp >= now - this.windowMs)
      const card = element('article', 'gpu-card')
      const cardHead = element('div', 'gpu-card-head')
      cardHead.append(element('strong', '', `GPU ${gpu.index}`), element('span', '', gpu.name))
      const metrics = element('div', 'gpu-metrics-grid')
      METRICS.forEach((metric) => {
        const chart = this.createMetricChart(metric, gpu, history, now)
        if (chart) metrics.append(chart)
      })
      const details = element('div', 'gpu-live-details')
      if (gpu.performance_state) details.append(element('span', '', t('gpu.state', { state: gpu.performance_state }, `State ${gpu.performance_state}`)))
      if (Number.isFinite(gpu.pcie_generation) && Number.isFinite(gpu.pcie_width)) details.append(element('span', '', `PCIe Gen ${gpu.pcie_generation} x${gpu.pcie_width}`))
      if (Number.isFinite(gpu.power_limit)) details.append(element('span', '', t('gpu.powerLimit', { power: Math.round(gpu.power_limit) }, `Power limit ${Math.round(gpu.power_limit)} W`)))
      card.append(cardHead, metrics, details)
      grid.append(card)
    })
    monitor.append(grid)
    this.output.replaceChildren(monitor)
  }

  async refresh() {
    if (this.refreshActive) return
    this.refreshActive = true
    try {
      const payload = await this.fetchJson('/system/gpu')
      this.stats = Array.isArray(payload.gpus) ? payload.gpus : []
      this.mergeHistory(payload.history)
      if (!this.hovering) this.render(this.stats)
    } catch {
      if (!this.hovering) this.render(this.stats)
    } finally {
      this.refreshActive = false
    }
  }

  start() {
    if (this.timer || document.hidden) return
    if (this.stats.length) this.render(this.stats)
    this.refresh()
    this.timer = setInterval(() => this.refresh(), POLL_INTERVAL_MS)
  }

  stop() {
    clearInterval(this.timer)
    this.timer = null
  }
}
