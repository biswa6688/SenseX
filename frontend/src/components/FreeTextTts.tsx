import { useMutation } from '@tanstack/react-query'
import { Loader2, Play } from 'lucide-react'
import { useState } from 'react'
import { api } from '../api/client'

export function FreeTextTts() {
  const [text, setText] = useState('')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const speak = useMutation({
    mutationFn: (t: string) => api.speak(t),
    onSuccess: (blob) => setAudioUrl(URL.createObjectURL(blob)),
  })

  return (
    <div className="rounded-xl border border-(--color-border) bg-(--color-bg-elevated) p-5">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type anything to hear it spoken aloud…"
        rows={3}
        className="w-full resize-none rounded-lg border border-(--color-border) bg-(--color-bg) p-3 text-sm outline-none focus:border-(--color-brand)"
      />
      <div className="mt-3 flex items-center gap-3">
        <button
          type="button"
          disabled={!text.trim() || speak.isPending}
          onClick={() => speak.mutate(text)}
          className="brand-gradient flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium text-(--color-brand-fg) disabled:opacity-40"
        >
          {speak.isPending ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} />}
          Speak
        </button>
        {speak.isError && (
          <span className="text-sm text-(--color-danger)">Failed: {String(speak.error)}</span>
        )}
        {audioUrl && <audio controls src={audioUrl} className="h-8" />}
      </div>
    </div>
  )
}
