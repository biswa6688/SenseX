interface LogoProps {
  size?: number
  animated?: boolean
  className?: string
}

/** Abstract neural-node AI mark: a central node with orbiting connected
 * nodes, currentColor-based so it adapts to theme. Hand-authored, not
 * sourced externally. */
export function Logo({ size = 40, animated = false, className = '' }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="SenseX"
    >
      <defs>
        <linearGradient id="sensex-logo-grad" x1="0" y1="0" x2="48" y2="48">
          <stop offset="0%" stopColor="var(--color-brand)" />
          <stop offset="100%" stopColor="var(--color-brand-2)" />
        </linearGradient>
      </defs>

      <g stroke="url(#sensex-logo-grad)" strokeWidth="1.5" opacity="0.7">
        <line x1="24" y1="24" x2="24" y2="6" />
        <line x1="24" y1="24" x2="39.8" y2="15" />
        <line x1="24" y1="24" x2="39.8" y2="33" />
        <line x1="24" y1="24" x2="24" y2="42" />
        <line x1="24" y1="24" x2="8.2" y2="33" />
        <line x1="24" y1="24" x2="8.2" y2="15" />
      </g>

      <circle cx="24" cy="24" r="7" fill="url(#sensex-logo-grad)">
        {animated && (
          <animate attributeName="r" values="7;8;7" dur="2.5s" repeatCount="indefinite" />
        )}
      </circle>

      {[
        [24, 6],
        [39.8, 15],
        [39.8, 33],
        [24, 42],
        [8.2, 33],
        [8.2, 15],
      ].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="3.2" fill="url(#sensex-logo-grad)">
          {animated && (
            <animate
              attributeName="opacity"
              values="0.5;1;0.5"
              dur="2.5s"
              begin={`${i * 0.2}s`}
              repeatCount="indefinite"
            />
          )}
        </circle>
      ))}
    </svg>
  )
}
