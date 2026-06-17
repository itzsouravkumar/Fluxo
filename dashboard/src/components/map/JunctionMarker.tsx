interface JunctionMarkerProps {
  junctionId: string
  name: string
  densityScore: number
  lat: number
  lng: number
}

export function JunctionMarker({ name, densityScore }: JunctionMarkerProps) {
  const color = densityScore > 0.7 ? 'bg-fluxo-critical' : densityScore > 0.4 ? 'bg-fluxo-yellow' : 'bg-fluxo-green'
  return (
    <div className={`p-2 rounded ${color} text-white text-xs`}>
      {name}: {(densityScore * 100).toFixed(0)}%
    </div>
  )
}
