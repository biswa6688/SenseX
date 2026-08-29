import { motion } from 'framer-motion'
import {
  AudioWaveform,
  Gauge,
  Mic,
  ScrollText,
  Smile,
  Users,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { Logo } from '../components/Logo'

const FEATURES = [
  {
    icon: ScrollText,
    title: 'Transcript',
    desc: 'Word-level accurate speech-to-text via faster-whisper, running entirely on CPU.',
  },
  {
    icon: Users,
    title: 'Speaker Diarization',
    desc: 'Who-said-what, turn by turn, merged onto the transcript at word-level precision.',
  },
  {
    icon: AudioWaveform,
    title: 'Text-to-Speech',
    desc: 'Natural Piper voices read summaries back to you, or speak any text you give it.',
  },
  {
    icon: ScrollText,
    title: 'Summary',
    desc: 'A local LLM condenses long calls into a clear, structured summary.',
  },
  {
    icon: Smile,
    title: 'Sentiment',
    desc: 'Overall and per-speaker sentiment, extracted straight from the conversation.',
  },
  {
    icon: Gauge,
    title: 'QA Ratings',
    desc: 'Rubric-scored quality assurance: greeting, empathy, resolution, compliance, and more.',
  },
]

export function Landing() {
  return (
    <div className="mx-auto max-w-6xl px-6">
      <section className="flex flex-col items-center gap-6 py-24 text-center">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Logo size={64} animated />
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl"
        >
          Understand every conversation,{' '}
          <span className="brand-gradient-text">without a GPU.</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="max-w-2xl text-lg text-(--color-fg-muted)"
        >
          Transcript, speaker diarization, summary, sentiment, and QA ratings — one local
          pipeline, tuned to run entirely on CPU in 16GB of RAM.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="flex gap-3"
        >
          <Link
            to="/playground"
            className="brand-gradient flex items-center gap-2 rounded-full px-6 py-3 font-medium text-(--color-brand-fg) shadow-lg shadow-(--color-brand)/20 transition-transform hover:scale-105"
          >
            <Mic size={18} />
            Try the Playground
          </Link>
          <Link
            to="/models"
            className="rounded-full border border-(--color-border) px-6 py-3 font-medium transition-colors hover:bg-(--color-bg-elevated)"
          >
            Download models
          </Link>
        </motion.div>
      </section>

      <section className="grid grid-cols-1 gap-4 pb-24 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, desc }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: i * 0.05 }}
            className="rounded-2xl border border-(--color-border) bg-(--color-bg-elevated) p-6"
          >
            <div className="brand-gradient mb-4 flex size-10 items-center justify-center rounded-xl text-(--color-brand-fg)">
              <Icon size={20} />
            </div>
            <h3 className="mb-1 font-semibold">{title}</h3>
            <p className="text-sm text-(--color-fg-muted)">{desc}</p>
          </motion.div>
        ))}
      </section>
    </div>
  )
}
