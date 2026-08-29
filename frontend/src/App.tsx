import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { SplashGate } from './components/SplashGate'
import { Jobs } from './pages/Jobs'
import { Landing } from './pages/Landing'
import { Models } from './pages/Models'
import { OpenCodeCli } from './pages/OpenCodeCli'
import { Playground } from './pages/Playground'

const queryClient = new QueryClient()

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SplashGate>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Landing />} />
              <Route path="playground" element={<Playground />} />
              <Route path="jobs" element={<Jobs />} />
              <Route path="models" element={<Models />} />
              <Route path="opencode-cli" element={<OpenCodeCli />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </SplashGate>
    </QueryClientProvider>
  )
}
