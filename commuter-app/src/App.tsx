import { useState, useEffect, useCallback } from 'react'

const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : `http://${window.location.hostname}:8000`

interface Alert {
  id: string
  type: string
  junction?: string
  severity: string
  plate?: string
  timestamp?: number
  density?: number
  level?: string
}

interface Junction {
  junction_id: string
  name: string
  density_score: number
  congestion_level: string
  lat: number
  lng: number
}

function useApi<T>(path: string, refreshMs: number = 30000): T | null {
  const [data, setData] = useState<T | null>(null)

  const fetcher = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}${path}`)
      const json = await res.json()
      setData(json)
    } catch {
      // offline fallback
    }
  }, [path])

  useEffect(() => {
    fetcher()
    const interval = setInterval(fetcher, refreshMs)
    return () => clearInterval(interval)
  }, [fetcher, refreshMs])

  return data
}

function Header() {
  return (
    <header className="bg-fluxo-card border-b border-fluxo-border px-4 py-3 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center gap-2">
        <h1 className="text-lg font-bold">FLUXO</h1>
        <span className="px-2 py-0.5 bg-fluxo-green/20 text-fluxo-green text-xs rounded-full">LIVE</span>
      </div>
      <div className="text-xs text-gray-400">Bengaluru Traffic</div>
    </header>
  )
}

function AlertCard({ alert }: { alert: Alert }) {
  const severityColor = (s: string) => {
    if (s === 'high') return 'border-fluxo-red bg-fluxo-red/5'
    if (s === 'medium') return 'border-fluxo-yellow bg-fluxo-yellow/5'
    return 'border-fluxo-green bg-fluxo-green/5'
  }

  const typeIcon = (t: string) => {
    switch (t) {
      case 'signal_jump': return '🚦'
      case 'no_helmet': return '🪖'
      case 'wrong_way': return '⚠️'
      case 'congestion': return '🚗'
      default: return '📍'
    }
  }

  const typeLabel = (t: string) => t.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')

  return (
    <div className={`border-l-4 rounded-r-lg p-3 ${severityColor(alert.severity)}`}>
      <div className="flex items-center gap-2">
        <span className="text-lg">{typeIcon(alert.type)}</span>
        <div className="flex-1">
          <div className="text-sm font-medium">{typeLabel(alert.type)}</div>
          {alert.junction && <div className="text-xs text-gray-400">{alert.junction}</div>}
          {alert.plate && <div className="text-xs text-gray-500 font-mono">{alert.plate}</div>}
        </div>
        {alert.density !== undefined && (
          <div className="text-right">
            <div className={`text-sm font-bold ${alert.density > 0.7 ? 'text-fluxo-red' : alert.density > 0.4 ? 'text-fluxo-yellow' : 'text-fluxo-green'}`}>
              {(alert.density * 100).toFixed(0)}%
            </div>
            {alert.level && <div className="text-xs text-gray-500">{alert.level}</div>}
          </div>
        )}
      </div>
    </div>
  )
}

function CongestionCard({ junction }: { junction: Junction }) {
  const color = junction.density_score > 0.7 ? 'text-fluxo-red' : junction.density_score > 0.4 ? 'text-fluxo-yellow' : 'text-fluxo-green'
  const bgColor = junction.density_score > 0.7 ? 'bg-fluxo-red/10' : junction.density_score > 0.4 ? 'bg-fluxo-yellow/10' : 'bg-fluxo-green/10'

  return (
    <div className={`rounded-lg p-3 ${bgColor} border border-fluxo-border`}>
      <div className="flex justify-between items-center">
        <div>
          <div className="text-sm font-medium">{junction.name}</div>
          <div className="text-xs text-gray-400">{junction.congestion_level}</div>
        </div>
        <div className={`text-xl font-bold ${color}`}>
          {(junction.density_score * 100).toFixed(0)}%
        </div>
      </div>
      <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${junction.density_score > 0.7 ? 'bg-fluxo-red' : junction.density_score > 0.4 ? 'bg-fluxo-yellow' : 'bg-fluxo-green'}`}
          style={{ width: `${junction.density_score * 100}%` }}
        />
      </div>
    </div>
  )
}

function DeparturePlanner() {
  const data = useApi<any>('/api/v1/commuter/departure', 60000)

  return (
    <div className="bg-fluxo-card rounded-lg border border-fluxo-border p-4">
      <h3 className="text-sm font-semibold mb-3">Best Times to Travel</h3>
      {data?.best_departures?.map((d: any, i: number) => (
        <div key={i} className="flex justify-between items-center py-2 border-b border-fluxo-border last:border-0">
          <div>
            <div className="text-sm font-mono text-fluxo-green">{d.time}</div>
            <div className="text-xs text-gray-400">{d.reason}</div>
          </div>
          <span className={`text-xs px-2 py-0.5 rounded ${d.congestion === 'low' ? 'bg-fluxo-green/20 text-fluxo-green' : 'bg-fluxo-yellow/20 text-fluxo-yellow'}`}>
            {d.congestion}
          </span>
        </div>
      ))}
    </div>
  )
}

function App() {
  const [tab, setTab] = useState<'alerts' | 'congestion' | 'departure'>('alerts')
  const alertsData = useApi<{ alerts: Alert[] }>('/api/v1/commuter/alerts?limit=20', 15000)
  const congestionData = useApi<{ junctions: Junction[] }>('/api/v1/commuter/congestion', 15000)

  const alerts = alertsData?.alerts || [
    { id: '1', type: 'signal_jump', junction: 'Veerannapalya', severity: 'high', plate: 'KA-09-HB-7823', timestamp: Date.now() / 1000 - 30 },
    { id: '2', type: 'no_helmet', junction: 'Silk Board', severity: 'medium', plate: 'KA-05-MJ-4421', timestamp: Date.now() / 1000 - 60 },
    { id: '3', type: 'congestion', junction: 'Hebbal', severity: 'high', density: 0.85, level: 'CRITICAL' },
  ]

  const junctions = congestionData?.junctions || [
    { junction_id: 'j1', name: 'Veerannapalya', density_score: 0.72, congestion_level: 'HIGH', lat: 12.998, lng: 77.689 },
    { junction_id: 'j2', name: 'Gokaldas', density_score: 0.45, congestion_level: 'MODERATE', lat: 12.995, lng: 77.683 },
    { junction_id: 'j3', name: 'Silk Board', density_score: 0.89, congestion_level: 'CRITICAL', lat: 12.918, lng: 77.621 },
    { junction_id: 'j4', name: 'Hebbal', density_score: 0.31, congestion_level: 'CLEAR', lat: 13.035, lng: 77.597 },
  ]

  return (
    <div className="min-h-screen bg-fluxo-bg pb-20">
      <Header />

      {/* Tab Navigation */}
      <div className="flex border-b border-fluxo-border bg-fluxo-card">
        {(['alerts', 'congestion', 'departure'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-3 text-sm font-medium capitalize ${
              tab === t ? 'text-fluxo-green border-b-2 border-fluxo-green' : 'text-gray-500'
            }`}
          >
            {t === 'alerts' ? 'Alerts' : t === 'congestion' ? 'Live Map' : 'Plan Trip'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-4">
        {tab === 'alerts' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-gray-400">RECENT ALERTS</h2>
              <span className="text-xs text-gray-500">{alerts.length} alerts</span>
            </div>
            {alerts.map(alert => (
              <AlertCard key={alert.id} alert={alert} />
            ))}
          </div>
        )}

        {tab === 'congestion' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-gray-400">CONGESTION MAP</h2>
              <span className="text-xs text-gray-500">{junctions.length} junctions</span>
            </div>
            {junctions.map(j => (
              <CongestionCard key={j.junction_id} junction={j} />
            ))}
          </div>
        )}

        {tab === 'departure' && <DeparturePlanner />}
      </div>

      {/* Bottom Nav */}
      <div className="fixed bottom-0 left-0 right-0 bg-fluxo-card border-t border-fluxo-border flex">
        {[
          { icon: '🔔', label: 'Alerts', key: 'alerts' },
          { icon: '🗺️', label: 'Map', key: 'congestion' },
          { icon: '🕐', label: 'Plan', key: 'departure' },
        ].map(item => (
          <button
            key={item.key}
            onClick={() => setTab(item.key as any)}
            className={`flex-1 py-3 flex flex-col items-center gap-1 ${
              tab === item.key ? 'text-fluxo-green' : 'text-gray-500'
            }`}
          >
            <span className="text-lg">{item.icon}</span>
            <span className="text-xs">{item.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default App
