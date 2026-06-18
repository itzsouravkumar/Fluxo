import { useEffect, useRef, useState, useCallback } from 'react'

interface UseWebSocketOptions {
  url: string
  enabled?: boolean
  onMessage?: (data: any) => void
}

export function useWebSocket({ url, enabled = true, onMessage }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<any>(null)

  useEffect(() => {
    if (!enabled) return

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setLastMessage(data)
        onMessage?.(data)
      } catch {
        setLastMessage(event.data)
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [url, enabled])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { connected, lastMessage, send }
}

export function useFluxoData() {
  const [junctions, setJunctions] = useState<Record<string, any>>({})
  const [violations, setViolations] = useState<any[]>([])
  const [alerts, setAlerts] = useState<any[]>([])
  const [connected, setConnected] = useState(false)

  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${wsProtocol}//${window.location.host}/ws/junctions`
  const { connected: wsConnected, lastMessage } = useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      if (data && typeof data === 'object') {
        setJunctions(data)
      }
    },
  })

  useEffect(() => {
    setConnected(wsConnected)
  }, [wsConnected])

  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const [jRes, vRes, aRes] = await Promise.all([
          fetch('/api/v1/junctions'),
          fetch('/api/v1/violations?limit=20'),
          fetch('/api/v1/commuter/alerts?limit=10'),
        ])
        const jData = await jRes.json()
        const vData = await vRes.json()
        const aData = await aRes.json()
        if (jData.junctions) setJunctions(jData.junctions)
        if (vData.violations) setViolations(vData.violations)
        if (aData.alerts) setAlerts(aData.alerts)
      } catch {
        // API not available, use mock data
        setJunctions({
          j1: { junction_id: 'j1', name: 'Veerannapalya', density_score: 0.72, congestion_level: 'HIGH', signal_phase: 'N-S', signal_remaining: 23, vehicle_count: 42, unique_vehicles: 38, lane_states: {}, violations: [], prediction: {} },
          j2: { junction_id: 'j2', name: 'Gokaldas', density_score: 0.45, congestion_level: 'MODERATE', signal_phase: 'E-W', signal_remaining: 15, vehicle_count: 28, unique_vehicles: 25, lane_states: {}, violations: [], prediction: {} },
          j3: { junction_id: 'j3', name: 'Silk Board', density_score: 0.89, congestion_level: 'CRITICAL', signal_phase: 'N-S', signal_remaining: 8, vehicle_count: 67, unique_vehicles: 58, lane_states: {}, violations: [], prediction: {} },
          j4: { junction_id: 'j4', name: 'Hebbal', density_score: 0.31, congestion_level: 'CLEAR', signal_phase: 'E-W', signal_remaining: 30, vehicle_count: 15, unique_vehicles: 14, lane_states: {}, violations: [], prediction: {} },
        })
        setViolations([
          { id: 'v1', type: 'no_helmet', junction_id: 'j1', plate_number: 'KA-05-MJ-4421', confidence: 0.92, timestamp: Date.now() / 1000 - 60 },
          { id: 'v2', type: 'signal_jump', junction_id: 'j3', plate_number: 'KA-09-HB-7823', confidence: 0.88, timestamp: Date.now() / 1000 - 30 },
          { id: 'v3', type: 'wrong_way', junction_id: 'j2', plate_number: 'KA-51-NA-2291', confidence: 0.95, timestamp: Date.now() / 1000 - 15 },
        ])
      }
    }
    fetchInitial()
  }, [])

  return { junctions, violations, alerts, connected }
}
