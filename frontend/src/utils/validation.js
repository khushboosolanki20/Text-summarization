import { formatBytes } from './format'

/**
 * Check an uploaded file's type and size in the browser before uploading
 * (the server validates again). Returns an error message, or null if OK.
 */
export function validateFile(file, config) {
  const extension = `.${file.name.split('.').pop()?.toLowerCase()}`
  if (extension === '.doc') return 'Legacy .doc files are not supported. Please save the document as .docx.'
  if (!config.supported_file_types.includes(extension)) {
    return `Unsupported file type "${extension}". Please upload ${config.supported_file_types.join(', ')}.`
  }
  if (file.size === 0) return 'The file is empty.'
  if (file.size > config.max_upload_mb * 1024 * 1024) {
    return `The file is too large (${formatBytes(file.size)}). The limit is ${config.max_upload_mb} MB.`
  }
  return null
}
