export type JobStatusValue = 'queued' | 'processing' | 'completed' | 'failed'

export interface JobStatus {
  jobId: string
  status: JobStatusValue
  stage: string | null
  error: string | null
}

export interface DiarizedTurn {
  start: number
  end: number
  speaker: string
  text: string
}

export interface QaCriterion {
  name: string
  score: number
  rationale: string
}

export interface JobResult {
  transcript: DiarizedTurn[]
  summary: string
  sentiment: { overall: string; [key: string]: unknown }
  qaRatings: { overallScore: number; criteria: QaCriterion[] }
  summaryAudioPath: string
}

export interface ModelInfo {
  id: string
  name: string
  repo: string
  requiresAuth: boolean
  authNote?: string
  downloaded: boolean
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
  return res.json() as Promise<T>
}

export const api = {
  submitAudioJob: (file: File): Promise<{ jobId: string; status: string }> => {
    const form = new FormData()
    form.append('file', file)
    return fetch('/api/audio-jobs', { method: 'POST', body: form }).then(json)
  },

  jobStatus: (jobId: string): Promise<JobStatus> =>
    fetch(`/api/audio-jobs/${jobId}`).then(json),

  jobResult: (jobId: string): Promise<JobResult> =>
    fetch(`/api/audio-jobs/${jobId}/result`).then(json),

  jobHistory: (): Promise<Record<string, unknown>[]> => fetch('/api/audio-jobs').then(json),

  originalAudioUrl: (jobId: string) => `/api/audio-jobs/${jobId}/audio/original`,
  summaryAudioUrl: (jobId: string) => `/api/audio-jobs/${jobId}/audio/summary`,

  speak: async (text: string, voice?: string): Promise<Blob> => {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice }),
    })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
    return res.blob()
  },

  listModels: (): Promise<ModelInfo[]> => fetch('/api/models').then(json),

  downloadModel: (id: string): Promise<ModelInfo> =>
    fetch(`/api/models/${id}/download`, { method: 'POST' }).then(json),
}
