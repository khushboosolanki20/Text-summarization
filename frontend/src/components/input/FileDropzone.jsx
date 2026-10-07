import { useRef, useState } from 'react'
import { formatBytes } from '../../utils/format'
import { validateFile } from '../../utils/validation'
import Icon from '../ui/Icon'

export default function FileDropzone({ file, onFile, onError, config, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const accept = (candidate) => {
    if (!candidate) return
    const error = validateFile(candidate, config)
    if (error) {
      onError(error)
      onFile(null)
    } else {
      onError(null)
      onFile(candidate)
    }
  }

  if (file) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
        <div className="rounded-lg bg-indigo-100 p-2 text-indigo-700">
          <Icon name="file" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-slate-900">{file.name}</p>
          <p className="text-xs text-slate-500">{formatBytes(file.size)}</p>
        </div>
        <button
          type="button"
          onClick={() => onFile(null)}
          disabled={disabled}
          className="rounded-lg p-2 text-slate-500 hover:bg-slate-200 hover:text-slate-800"
          aria-label="Remove file"
        >
          <Icon name="x" />
        </button>
      </div>
    )
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload a document: drop a file here or press Enter to browse"
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (!disabled) accept(e.dataTransfer.files?.[0])
      }}
      className={`flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition ${
        dragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-indigo-50/40'
      }`}
    >
      <div className="rounded-full bg-white p-3 text-indigo-600 shadow-sm ring-1 ring-slate-200">
        <Icon name="upload" className="h-6 w-6" />
      </div>
      <div>
        <p className="font-medium text-slate-800">
          Drop a file here, or <span className="text-indigo-600">browse</span>
        </p>
        <p className="mt-1 text-sm text-slate-500">
          PDF, DOCX or TXT · up to {config.max_upload_mb} MB · scanned PDFs are not supported
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={config.supported_file_types.join(',')}
        onChange={(e) => {
          accept(e.target.files?.[0])
          e.target.value = ''
        }}
      />
    </div>
  )
}
