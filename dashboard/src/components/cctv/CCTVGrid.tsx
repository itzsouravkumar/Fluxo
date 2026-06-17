export function CCTVGrid() {
  return (
    <div className="grid grid-cols-2 gap-2">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="aspect-video bg-gray-900 rounded flex items-center justify-center text-gray-500 text-sm">
          CAM {i}
        </div>
      ))}
    </div>
  )
}
