export function DiamondField({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 400 400"
      className={className}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect
        x="200"
        y="70"
        width="184"
        height="184"
        rx="6"
        transform="rotate(45 200 200)"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path
        d="M 60 330 A 260 260 0 0 1 340 330"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeDasharray="2 10"
        strokeLinecap="round"
      />
      <circle cx="200" cy="330" r="6" fill="currentColor" />
      <circle cx="330" cy="200" r="6" fill="currentColor" />
      <circle cx="200" cy="70" r="6" fill="currentColor" />
      <circle cx="70" cy="200" r="6" fill="currentColor" />
    </svg>
  );
}
