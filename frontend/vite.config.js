import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const proxyTarget = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:5000'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    watch: {
      usePolling: true,
      interval: 500
    },
    proxy: {
      '/api': proxyTarget
    }
  },
  test: {
    environment: 'jsdom'
  }
})
