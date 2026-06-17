import { useEffect, useRef, useState } from 'react'

export function useWebSocket(url: string) {
  const ws = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<string | null>(null)

  useEffect(() => {
    ws.current = new WebSocket(url)

    ws.current.onopen = () => setIsConnected(true)
    ws.current.onclose = () => setIsConnected(false)
    ws.current.onmessage = (event) => setLastMessage(event.data)

    return () => {
      ws.current?.close()
    }
  }, [url])

  const send = (data: string) => {
    ws.current?.send(data)
  }

  return { isConnected, lastMessage, send }
}
