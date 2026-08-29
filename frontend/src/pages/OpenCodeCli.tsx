import { ExternalLink, Terminal } from 'lucide-react'

const PLATFORMS = [
  { label: 'macOS / Linux', cmd: 'curl -fsSL https://opencode.ai/install | bash' },
  { label: 'Windows (PowerShell)', cmd: 'irm https://opencode.ai/install.ps1 | iex' },
]

export function OpenCodeCli() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8 flex items-center gap-3">
        <div className="brand-gradient flex size-12 items-center justify-center rounded-xl text-(--color-brand-fg)">
          <Terminal size={22} />
        </div>
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">OpenCode CLI</h1>
          <p className="text-(--color-fg-muted)">
            The open-source terminal agent — an independent project, not built by IntelliSense.
          </p>
        </div>
      </div>

      <div className="space-y-3">
        {PLATFORMS.map(({ label, cmd }) => (
          <div key={label} className="rounded-xl border border-(--color-border) bg-(--color-bg-elevated) p-5">
            <p className="mb-2 text-sm font-medium">{label}</p>
            <code className="block overflow-x-auto rounded-lg bg-(--color-bg) px-3 py-2 text-xs">{cmd}</code>
          </div>
        ))}
      </div>

      <a
        href="https://github.com/sst/opencode/releases"
        target="_blank"
        rel="noreferrer"
        className="brand-gradient mt-6 inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium text-(--color-brand-fg)"
      >
        View all releases <ExternalLink size={14} />
      </a>
    </div>
  )
}
