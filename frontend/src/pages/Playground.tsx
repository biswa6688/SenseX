import { useMutation, useQuery } from '@tanstack/react-query'
import { Ban, Check, Loader2, Speaker, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api, type JobStatus } from '../api/client'
import { FreeTextTts } from '../components/FreeTextTts'
import { ResultsView } from '../components/ResultsView'

/** Pushed job status via Server-Sent Events instead of REST polling — one
 * long-lived connection per job, server closes it once the job reaches a
 * terminal state (see ml-service routers/jobs.py stream_job_status). */
function useJobStatusStream(jobId: string | null): JobStatus | null {
  const [status, setStatus] = useState<JobStatus | null>(null)

  useEffect(() => {
    setStatus(null)
    if (!jobId) return

    const source = new EventSource(api.jobStatusStreamUrl(jobId))
    source.onmessage = (event) => {
      const data = JSON.parse(event.data) as JobStatus
      setStatus(data)
      if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
        source.close()
      }
    }
    source.onerror = () => {
      // Browser auto-retries a dropped connection on its own; nothing to do
      // here beyond leaving the last known status in place.
    }

    return () => source.close()
  }, [jobId])

  return status
}

const STAGE_LABELS: Record<string, string> = {
  staging: 'Staging audio',
  transcribing: 'Transcribing speech',
  diarizing: 'Identifying speakers',
  merging: 'Merging transcript',
  analyzing: 'Summarizing, scoring sentiment & QA',
  synthesizing: 'Generating summary audio',
}

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const m = Math.floor(total / 60)
  const s = total % 60
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function JobTimeline({
  status,
  onCancel,
  cancelling,
}: {
  status: JobStatus
  onCancel: () => void
  cancelling: boolean
}) {
  const stages = status.stages.length ? status.stages : Object.keys(STAGE_LABELS)
  const currentIndex = status.stage ? stages.indexOf(status.stage) : -1

  return (
    <div className="mt-6 rounded-xl border border-(--color-border) bg-(--color-bg-elevated) px-5 py-4">
      <div className="flex items-center justify-between gap-4">
        <span className="font-medium">
          {STAGE_LABELS[status.stage ?? ''] ?? 'Processing…'}
        </span>
        <div className="flex items-center gap-3 text-sm text-(--color-fg-muted)">
          {status.elapsedSeconds != null && <span>Elapsed {formatDuration(status.elapsedSeconds)}</span>}
          {status.etaSeconds != null ? (
            <span>~{formatDuration(status.etaSeconds)} left</span>
          ) : (
            <span>Estimating time left…</span>
          )}
          <button
            onClick={onCancel}
            disabled={cancelling}
            className="flex items-center gap-1.5 rounded-full border border-(--color-danger)/30 px-3 py-1 text-(--color-danger) transition-colors hover:bg-(--color-danger)/10 disabled:opacity-50"
          >
            {cancelling ? <Loader2 className="animate-spin" size={13} /> : <Ban size={13} />}
            Cancel
          </button>
        </div>
      </div>

      <ol className="mt-4 flex flex-wrap items-center gap-x-2 gap-y-3">
        {stages.map((stage, i) => {
          const isDone = currentIndex >= 0 && i < currentIndex
          const isCurrent = i === currentIndex
          return (
            <li key={stage} className="flex items-center gap-2">
              <div
                className={
                  'flex size-6 items-center justify-center rounded-full text-xs ' +
                  (isDone
                    ? 'brand-gradient text-(--color-brand-fg)'
                    : isCurrent
                      ? 'border-2 border-(--color-brand) text-(--color-brand)'
                      : 'border border-(--color-border) text-(--color-fg-muted)')
                }
              >
                {isDone ? (
                  <Check size={14} />
                ) : isCurrent ? (
                  <Loader2 className="animate-spin" size={12} />
                ) : (
                  i + 1
                )}
              </div>
              <span
                className={
                  isCurrent
                    ? 'text-sm font-medium'
                    : isDone
                      ? 'text-sm text-(--color-fg-muted)'
                      : 'text-sm text-(--color-fg-muted)/60'
                }
              >
                {STAGE_LABELS[stage] ?? stage}
              </span>
              {i < stages.length - 1 && (
                <span className="mx-1 h-px w-4 bg-(--color-border) sm:w-8" />
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}

export function Playground() {
  const [jobId, setJobId] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const submit = useMutation({
    mutationFn: (file: File) => api.submitAudioJob(file),
    onSuccess: (data) => setJobId(data.jobId),
  })

  const status = useJobStatusStream(jobId)

  const cancel = useMutation({
    mutationFn: () => api.cancelJob(jobId!),
  })

  const result = useQuery({
    queryKey: ['job-result', jobId],
    queryFn: () => api.jobResult(jobId!),
    enabled: status?.status === 'completed',
  })

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

      {status?.status === 'queued' && (
        <div className="mt-6 flex items-center justify-between gap-3 rounded-xl border border-(--color-border) bg-(--color-bg-elevated) px-5 py-4">
          <div className="flex items-center gap-3">
            <Loader2 className="animate-spin text-(--color-brand)" size={20} />
            <span>
              Queued — waiting for the current job to finish
              {status.queuePosition != null &&
                status.queueLength != null &&
                ` (position ${status.queuePosition} of ${status.queueLength})`}
            </span>
          </div>
          <button
            onClick={() => cancel.mutate()}
            disabled={cancel.isPending}
            className="flex shrink-0 items-center gap-1.5 rounded-full border border-(--color-danger)/30 px-3 py-1.5 text-sm font-medium text-(--color-danger) transition-colors hover:bg-(--color-danger)/10 disabled:opacity-50"
          >
            {cancel.isPending ? <Loader2 className="animate-spin" size={14} /> : <Ban size={14} />}
            Cancel
          </button>
        </div>
      )}

      {status?.status === 'processing' && (
        <JobTimeline status={status} onCancel={() => cancel.mutate()} cancelling={cancel.isPending} />
      )}

      {status?.status === 'failed' && (
        <p className="mt-6 rounded-xl border border-(--color-danger)/30 bg-(--color-danger)/10 px-5 py-4 text-(--color-danger)">
          Job failed: {status.error}
        </p>
      )}

      {status?.status === 'cancelled' && (
        <p className="mt-6 rounded-xl border border-(--color-warning)/30 bg-(--color-warning)/10 px-5 py-4 text-(--color-warning)">
          Job cancelled.
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
