import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useState, type ReactNode } from 'react'
import { Logo } from './Logo'

const SESSION_KEY = 'intellisense-splash-shown'
const SPLASH_MS = 5000

export function SplashGate({ children }: { children: ReactNode }) {
  const [showSplash, setShowSplash] = useState(() => !sessionStorage.getItem(SESSION_KEY))

  useEffect(() => {
    if (!showSplash) return
    const timer = setTimeout(() => {
      sessionStorage.setItem(SESSION_KEY, '1')
      setShowSplash(false)
    }, SPLASH_MS)
    return () => clearTimeout(timer)
  }, [showSplash])

  return (
    <>
      <AnimatePresence>
        {showSplash && (
          <motion.div
            className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-6 bg-(--color-bg)"
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div
              initial={{ scale: 0.6, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.7, ease: 'easeOut' }}
            >
              <Logo size={96} animated />
            </motion.div>
            <motion.h1
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="brand-gradient-text text-3xl font-semibold tracking-tight"
            >
              IntelliSense
            </motion.h1>
          </motion.div>
        )}
      </AnimatePresence>
      {!showSplash && children}
    </>
  )
}
