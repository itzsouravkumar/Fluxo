interface CameraFeedProps {
  cameraId: number
  streamUrl?: string
}

export function CameraFeed({ cameraId, streamUrl }: CameraFeedProps) {
  return (
    <div className="aspect-video bg-gray-900 rounded flex items-center justify-center text-gray-500 text-sm">
      {streamUrl ? (
        <video src={streamUrl} className="w-full h-full object-cover rounded" autoPlay muted />
      ) : (
        <span>CAM {cameraId}</span>
      )}
    </div>
  )
}
