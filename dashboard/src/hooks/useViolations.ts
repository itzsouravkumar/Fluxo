import { useQuery } from '@tanstack/react-query'

interface Violation {
  id: string
  type: string
  plate_number: string
  timestamp: string
  junction_id: string
}

export function useViolations(junctionId?: string) {
  return useQuery<Violation[]>({
    queryKey: ['violations', junctionId],
    queryFn: async () => {
      const params = junctionId ? `?junction_id=${junctionId}` : ''
      const res = await fetch(`/api/v1/violations/${params}`)
      const data = await res.json()
      return data.violations || []
    },
    refetchInterval: 3000,
  })
}
