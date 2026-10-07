import axios from 'axios'

// In development Vite proxies /api to the FastAPI server (see vite.config.js).
// For a production build, set VITE_API_BASE_URL to the backend origin.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 10 * 60 * 1000, // abstractive summarisation of long documents on CPU can be slow
})

/**
 * Convert any Axios error into a short, human-readable message.
 * The backend always returns `{ detail: string }` for handled errors.
 */
export function toErrorMessage(error) {
  if (error.response?.data?.detail) {
    const { detail } = error.response.data
    return typeof detail === 'string' ? detail : 'The request was invalid.'
  }
  if (error.code === 'ECONNABORTED') return 'The request timed out.'
  if (error.request) return 'Cannot reach the IntelliSum server. Is the backend running?'
  return 'Something went wrong. Please try again.'
}

export async function getHealth() {
  const { data } = await api.get('/api/health')
  return data
}

export default api
