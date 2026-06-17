interface ViolationClipProps {
  clipUrl: string
  violationType: string
  plateNumber: string
}

export function ViolationClip({ clipUrl, violationType, plateNumber }: ViolationClipProps) {
  return (
    <div className="bg-gray-900 rounded p-2">
      <video src={clipUrl} className="w-full rounded" controls />
      <div className="mt-2 text-xs text-gray-400">
        {violationType} | {plateNumber}
      </div>
    </div>
  )
}
