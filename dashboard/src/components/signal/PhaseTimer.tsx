interface PhaseTimerProps {
  elapsed: number
  remaining: number
}

export function PhaseTimer({ elapsed, remaining }: PhaseTimerProps) {
  return (
    <div className="font-mono text-sm">
      <span className="text-fluxo-green">{elapsed}s</span>
      <span className="text-gray-500"> / </span>
      <span className="text-gray-400">{remaining}s</span>
    </div>
  )
}
