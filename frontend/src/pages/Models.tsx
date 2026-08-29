import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, DownloadCloud, Loader2, ShieldAlert } from 'lucide-react'
import { api } from '../api/client'

export function Models() {
  const queryClient = useQueryClient()
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })

  const download = useMutation({
    mutationFn: (id: string) => api.downloadModel(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models'] }),
  })

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <h1 className="mb-2 text-3xl font-semibold tracking-tight">Models</h1>
      <p className="mb-8 text-(--color-fg-muted)">
        All inference runs locally on CPU. Download each model once — they're cached in{' '}
        <code className="rounded bg-(--color-border)/50 px-1">storage/models/</code>.
      </p>

      <div className="space-y-3">
        {models.isLoading && <Loader2 className="animate-spin" />}
        {models.data?.map((model) => (
          <div
            key={model.id}
            className="flex items-center justify-between rounded-xl border border-(--color-border) bg-(--color-bg-elevated) p-5"
          >
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{model.name}</span>
                {model.downloaded && <CheckCircle2 size={16} className="text-(--color-success)" />}
              </div>
              <p className="text-xs text-(--color-fg-muted)">{model.repo}</p>
              {model.requiresAuth && (
                <p className="mt-1 flex items-start gap-1.5 text-xs text-(--color-warning)">
                  <ShieldAlert size={14} className="mt-0.5 shrink-0" />
                  {model.authNote}
                </p>
              )}
            </div>

            {!model.downloaded && !model.requiresAuth && (
              <button
                onClick={() => download.mutate(model.id)}
                disabled={download.isPending && download.variables === model.id}
                className="brand-gradient flex shrink-0 items-center gap-2 rounded-full px-4 py-2 text-sm font-medium text-(--color-brand-fg) disabled:opacity-50"
              >
                {download.isPending && download.variables === model.id ? (
                  <Loader2 className="animate-spin" size={14} />
                ) : (
                  <DownloadCloud size={14} />
                )}
                Download
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
