import { useQuery } from '@tanstack/react-query'

interface Junction {
  id: string
  name: string
  density_score: number
  congestion_level: string
  lat: number
  lng: number
}

export function useJunctions() {
  return useQuery<Junction[]>({
    queryKey: ['junctions'],
    queryFn: async () => {
      const res = await fetch('/api/v1/junctions/')
      const data = await res.json()
      return data.junctions || []
    },
    refetchInterval: 5000,
  })
}
