import { useEffect, useRef } from 'react'

const MAPPLS_KEY = 'vahxjkzlgpxdvweyqgiftnipxfooloiqoqai'

declare global {
  interface Window {
    mappls: any
  }
}

interface Junction {
  junction_id: string
  name: string
  lat: number
  lng: number
  density_score: number
  congestion_level: string
}

export function FluxoMap({ junctions = {} }: { junctions?: Record<string, any> }) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstance = useRef<any>(null)

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return

    const script = document.createElement('script')
    script.src = `https://apis.mappls.com/advancedmaps/api/${MAPPLS_KEY}/map_sdk?v=3.0&layer=vector`
    script.async = true
    script.onload = () => {
      if (window.mappls && mapRef.current) {
        mapInstance.current = new window.mappls.Map(mapRef.current, {
          center: { lat: 12.97, lng: 77.59 },
          zoom: 12,
        })
        addMarkers()
      }
    }
    document.head.appendChild(script)

    return () => {
      document.head.removeChild(script)
    }
  }, [])

  useEffect(() => {
    if (mapInstance.current) {
      addMarkers()
    }
  }, [junctions])

  function addMarkers() {
    if (!mapInstance.current || !window.mappls) return

    const jList = Object.values(junctions) as Junction[]
    jList.forEach((j) => {
      const color = j.density_score > 0.7 ? 'red' : j.density_score > 0.4 ? 'yellow' : 'green'
      const marker = new window.mappls.Marker({
        map: mapInstance.current,
        position: { lat: j.lat, lng: j.lng },
        popupHtml: `<div style="padding:8px;min-width:150px">
          <b>${j.name}</b><br/>
          <span style="color:${color}">${j.congestion_level} (${(j.density_score * 100).toFixed(0)}%)</span>
        </div>`,
      })
    })
  }

  return (
    <div
      ref={mapRef}
      className="w-full h-96 rounded bg-gray-900"
      style={{ minHeight: '400px' }}
    />
  )
}
