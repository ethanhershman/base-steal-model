export function DiamondMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      className={className}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect
        x="16"
        y="3"
        width="18.4"
        height="18.4"
        rx="3"
        transform="rotate(45 16 16)"
        className="fill-brand-navy"
      />
      <rect
        x="16"
        y="9"
        width="9.5"
        height="9.5"
        rx="1.5"
        transform="rotate(45 16 16)"
        className="fill-brand-red"
      />
    </svg>
  );
}
