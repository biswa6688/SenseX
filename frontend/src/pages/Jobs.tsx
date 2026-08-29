import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Ban, Loader2 } from 'lucide-react'
import type { JobHistoryEntry, JobStatusValue } from '../api/client'
import { api } from '../api/client'

const STATUS_STYLE: Record<JobStatusValue, string> = {
  queued: 'bg-(--color-fg-muted)/10 text-(--color-fg-muted)',
  processing: 'bg-(--color-brand)/10 text-(--color-brand)',
  completed: 'bg-(--color-success)/10 text-(--color-success)',
  failed: 'bg-(--color-danger)/10 text-(--color-danger)',
  cancelled: 'bg-(--color-warning)/10 text-(--color-warning)',
}

const STAGE_LABELS: Record<string, string> = {
  staging: 'Staging audio',
  transcribing: 'Transcribing speech',
  diarizing: 'Identifying speakers',
  merging: 'Merging transcript',
  analyzing: 'Summarizing, scoring sentiment & QA',
  synthesizing: 'Generating summary audio',
}

function summarizeError(error: string): string {
  const firstLine = error.split('\n')[0].trim()
  return firstLine.length > 160 ? `${firstLine.slice(0, 160)}…` : firstLine
}

function formatWhen(epochSeconds: number): string {
  const diffMs = Date.now() - epochSeconds * 1000
  const diffSec = Math.max(0, Math.round(diffMs / 1000))
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.round(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.round(diffMin / 60)
  return `${diffHr}h ago`
}

export function Jobs() {
  const queryClient = useQueryClient()

  const jobs = useQuery({
    queryKey: ['job-history'],
    queryFn: api.jobHistory,
    // Secondary monitoring view (not the primary per-job progress UI, which
    // uses SSE — see Playground.tsx). A short poll here is simple and cheap
    // for an all-jobs dashboard rather than multiplexing many SSE streams.
    refetchInterval: 3000,
  })

  const cancel = useMutation({
    mutationFn: (id: string) => api.cancelJob(id),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['job-history'] }),
  })

  const active = (jobs.data ?? []).filter((j) => j.status === 'queued' || j.status === 'processing')
  const others = (jobs.data ?? []).filter((j) => j.status !== 'queued' && j.status !== 'processing')

  function renderRow(job: JobHistoryEntry) {
    const isActive = job.status === 'queued' || job.status === 'processing'
    return (
      <div
        key={job.id}
        className="flex items-center justify-between gap-4 rounded-xl border border-(--color-border) bg-(--color-bg-elevated) p-4"
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate font-mono text-xs text-(--color-fg-muted)">{job.id}</span>
            <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium capitalize ${STATUS_STYLE[job.status]}`}>
              {job.status}
            </span>
          </div>
          <p className="mt-1 truncate text-sm" title={job.status === 'failed' ? job.error ?? undefined : undefined}>
            {job.status === 'failed' && job.error
              ? summarizeError(job.error)
              : job.stage
                ? (STAGE_LABELS[job.stage] ?? job.stage)
                : '—'}
          </p>
          <p className="mt-0.5 text-xs text-(--color-fg-muted)">
            created {formatWhen(job.createdAt)} · updated {formatWhen(job.updatedAt)}
          </p>
        </div>

        {isActive && (
          <button
            onClick={() => cancel.mutate(job.id)}
            disabled={cancel.isPending && cancel.variables === job.id}
            className="flex shrink-0 items-center gap-1.5 rounded-full border border-(--color-danger)/30 px-3 py-1.5 text-sm font-medium text-(--color-danger) transition-colors hover:bg-(--color-danger)/10 disabled:opacity-50"
          >
            {cancel.isPending && cancel.variables === job.id ? (
              <Loader2 className="animate-spin" size={14} />
            ) : (
              <Ban size={14} />
            )}
            Cancel
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <h1 className="mb-2 text-3xl font-semibold tracking-tight">Jobs</h1>
      <p className="mb-8 text-(--color-fg-muted)">
        Every audio-pipeline job ever submitted — the queue runs one job at a time (see{' '}
        <code className="rounded bg-(--color-border)/50 px-1">DECISIONS.md</code> #9), so anything queued behind an
        active job shows its position here rather than in the Playground.
      </p>

      {jobs.isLoading && <Loader2 className="animate-spin" />}

      {jobs.data && jobs.data.length === 0 && (
        <p className="text-(--color-fg-muted)">No jobs yet — submit one from the Playground.</p>
      )}

      {active.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-(--color-fg-muted)">
            Active ({active.length})
          </h2>
          <div className="space-y-3">{active.map(renderRow)}</div>
        </div>
      )}

      {others.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-(--color-fg-muted)">History</h2>
          <div className="space-y-3">{others.map(renderRow)}</div>
        </div>
      )}
    </div>
  )
}
