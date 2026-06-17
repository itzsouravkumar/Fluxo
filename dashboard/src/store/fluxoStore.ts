import { create } from 'zustand'

interface Junction {
  id: string
  name: string
  density_score: number
  congestion_level: string
  lat: number
  lng: number
}

interface FluxoState {
  junctions: Junction[]
  selectedJunction: string | null
  setJunctions: (junctions: Junction[]) => void
  selectJunction: (id: string | null) => void
}

export const useJunctionStore = create<FluxoState>((set) => ({
  junctions: [],
  selectedJunction: null,
  setJunctions: (junctions) => set({ junctions }),
  selectJunction: (id) => set({ selectedJunction: id }),
}))
