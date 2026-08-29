import { AlertTriangle } from 'lucide-react'
import { useState } from 'react'
import type { JobResult } from '../api/client'
import { api } from '../api/client'

type Tab = 'transcript' | 'diarization' | 'summary' | 'sentiment' | 'qa'

const TABS: { id: Tab; label: string }[] = [
  { id: 'transcript', label: 'Transcript' },
  { id: 'diarization', label: 'Diarization' },
  { id: 'summary', label: 'Summary' },
  { id: 'sentiment', label: 'Sentiment' },
  { id: 'qa', label: 'QA Ratings' },
]

const SPEAKER_COLORS = ['var(--color-brand)', 'var(--color-accent)', 'var(--color-brand-2)', 'var(--color-warning)']

function speakerColor(speaker: string, speakers: string[]) {
  return SPEAKER_COLORS[speakers.indexOf(speaker) % SPEAKER_COLORS.length]
}

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function ResultsView({ jobId, result }: { jobId: string; result: JobResult }) {
  const [tab, setTab] = useState<Tab>('transcript')
  const speakers = [...new Set(result.transcript.map((t) => t.speaker))]

  const talkTime: Record<string, number> = {}
  for (const turn of result.transcript) {
    talkTime[turn.speaker] = (talkTime[turn.speaker] ?? 0) + (turn.end - turn.start)
  }
  const totalTime = Object.values(talkTime).reduce((a, b) => a + b, 0) || 1

  return (
    <div className="mt-10 rounded-2xl border border-(--color-border) bg-(--color-bg-elevated)">
      <div className="flex gap-1 border-b border-(--color-border) px-4 pt-3">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`rounded-t-lg px-4 py-2 text-sm font-medium transition-colors ${
              tab === id
                ? 'border-b-2 border-(--color-brand) text-(--color-brand)'
                : 'text-(--color-fg-muted) hover:text-(--color-fg)'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="p-6">
        {tab === 'transcript' && (
          <div className="space-y-3">
            {result.transcript.map((turn, i) => (
              <div key={i} className="flex gap-3">
                <span className="w-12 shrink-0 pt-0.5 text-xs text-(--color-fg-muted)">
                  {formatTime(turn.start)}
                </span>
                <div>
                  <span className="flex items-center gap-1.5">
                    <span
                      className="text-xs font-semibold"
                      style={{ color: speakerColor(turn.speaker, speakers) }}
                    >
                      {turn.speaker}
                    </span>
                    {turn.uncertain && (
                      <span
                        className="flex items-center gap-1 text-xs text-(--color-warning)"
                        title="This turn is long and content-dense enough that a speaker change may have been missed — the whole span could actually be two speakers."
                      >
                        <AlertTriangle size={12} />
                        uncertain
                      </span>
                    )}
                  </span>
                  <p className="text-sm">{turn.text}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {tab === 'diarization' && (
          <div className="space-y-4">
            <p className="text-sm text-(--color-fg-muted)">Talk time by speaker</p>
            {speakers.map((speaker) => (
              <div key={speaker}>
                <div className="mb-1 flex justify-between text-sm">
                  <span style={{ color: speakerColor(speaker, speakers) }} className="font-medium">
                    {speaker}
                  </span>
                  <span className="text-(--color-fg-muted)">
                    {formatTime(talkTime[speaker] ?? 0)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-(--color-border)">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${((talkTime[speaker] ?? 0) / totalTime) * 100}%`,
                      background: speakerColor(speaker, speakers),
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {tab === 'summary' && (
          <div className="space-y-4">
            <p className="text-sm leading-relaxed">{result.summary}</p>
            <audio controls src={api.summaryAudioUrl(jobId)} className="w-full" />
          </div>
        )}

        {tab === 'sentiment' && (
          <div className="space-y-2 text-sm">
            {Object.entries(result.sentiment).map(([key, value]) => (
              <div key={key} className="flex justify-between border-b border-(--color-border) py-2">
                <span className="capitalize text-(--color-fg-muted)">{key}</span>
                <span className="font-medium">{String(value)}</span>
              </div>
            ))}
          </div>
        )}

        {tab === 'qa' && (
          <div className="space-y-5">
            <div className="flex items-center gap-3">
              <span className="text-3xl font-semibold brand-gradient-text">
                {result.qaRatings.overallScore}
              </span>
              <span className="text-(--color-fg-muted)">/ 10 overall</span>
            </div>
            {result.qaRatings.criteria.map((c) => (
              <div key={c.name}>
                <div className="mb-1 flex justify-between text-sm">
                  <span className="font-medium">{c.name}</span>
                  <span className="text-(--color-fg-muted)">{c.score}/10</span>
                </div>
                <div className="mb-1 h-1.5 overflow-hidden rounded-full bg-(--color-border)">
                  <div className="brand-gradient h-full rounded-full" style={{ width: `${c.score * 10}%` }} />
                </div>
                <p className="text-xs text-(--color-fg-muted)">{c.rationale}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
