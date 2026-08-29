import { useMutation, useQuery } from '@tanstack/react-query'
import { Loader2, Speaker, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { api } from '../api/client'
import { FreeTextTts } from '../components/FreeTextTts'
import { ResultsView } from '../components/ResultsView'

const STAGE_LABELS: Record<string, string> = {
  staging: 'Staging audio',
  transcribing: 'Transcribing speech',
  diarizing: 'Identifying speakers',
  merging: 'Merging transcript',
  analyzing: 'Summarizing, scoring sentiment & QA',
  synthesizing: 'Generating summary audio',
}

export function Playground() {
  const [jobId, setJobId] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const submit = useMutation({
    mutationFn: (file: File) => api.submitAudioJob(file),
    onSuccess: (data) => setJobId(data.jobId),
  })

  const status = useQuery({
    queryKey: ['job-status', jobId],
    queryFn: () => api.jobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const s = query.state.data?.status
      return s === 'completed' || s === 'failed' ? false : 1500
    },
  })

  const result = useQuery({
    queryKey: ['job-result', jobId],
    queryFn: () => api.jobResult(jobId!),
    enabled: status.data?.status === 'completed',
  })

  const isRunning = !!jobId && status.data && status.data.status !== 'completed' && status.data.status !== 'failed'

  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <h1 className="mb-2 text-3xl font-semibold tracking-tight">Playground</h1>
      <p className="mb-8 text-(--color-fg-muted)">
        Upload a call recording to run the full pipeline: transcript, diarization, summary,
        sentiment, and QA ratings.
      </p>

      <div
        className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-(--color-border) bg-(--color-bg-elevated) px-6 py-16 text-center transition-colors hover:border-(--color-brand)"
        onClick={() => fileInput.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          const file = e.dataTransfer.files[0]
          if (file) submit.mutate(file)
        }}
      >
        <div className="brand-gradient flex size-12 items-center justify-center rounded-full text-(--color-brand-fg)">
          <Upload size={22} />
        </div>
        <p className="font-medium">Drop an audio file here, or click to browse</p>
        <p className="text-sm text-(--color-fg-muted)">wav, mp3, m4a, webm, ogg</p>
        <input
          ref={fileInput}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) submit.mutate(file)
          }}
        />
      </div>

      {submit.isPending && (
        <p className="mt-4 flex items-center gap-2 text-(--color-fg-muted)">
          <Loader2 className="animate-spin" size={16} /> Uploading…
        </p>
      )}
      {submit.isError && (
        <p className="mt-4 text-(--color-danger)">Upload failed: {String(submit.error)}</p>
      )}

      {isRunning && (
        <div className="mt-6 flex items-center gap-3 rounded-xl border border-(--color-border) bg-(--color-bg-elevated) px-5 py-4">
          <Loader2 className="animate-spin text-(--color-brand)" size={20} />
          <span>{STAGE_LABELS[status.data?.stage ?? ''] ?? 'Processing…'}</span>
        </div>
      )}

      {status.data?.status === 'failed' && (
        <p className="mt-6 rounded-xl border border-(--color-danger)/30 bg-(--color-danger)/10 px-5 py-4 text-(--color-danger)">
          Job failed: {status.data.error}
        </p>
      )}

      {result.data && jobId && <ResultsView jobId={jobId} result={result.data} />}

      <div className="mt-16 border-t border-(--color-border) pt-10">
        <h2 className="mb-1 flex items-center gap-2 text-xl font-semibold">
          <Speaker size={20} /> Free-text speech
        </h2>
        <p className="mb-4 text-sm text-(--color-fg-muted)">
          Standalone text-to-speech, independent of the pipeline above.
        </p>
        <FreeTextTts />
      </div>
    </div>
  )
}
