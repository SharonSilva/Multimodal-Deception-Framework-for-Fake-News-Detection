import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict':     'http://localhost:5001',
      '/batch':       'http://localhost:5001',
      '/describe':    'http://localhost:5001',
      '/ai_describe': 'http://localhost:5001',
      '/health':      'http://localhost:5001',
    }
  }
})
