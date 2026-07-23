import { NavLink, useLocation } from 'react-router';
import { AnimatedBackground } from '@/components/ui/animated-background';
import { DiamondMark } from '@/components/graphics/DiamondMark';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { to: '/', label: 'Home' },
  { to: '/predictor', label: 'Predictor' },
  { to: '/model-performance', label: 'Model Performance' },
  { to: '/about', label: 'About' },
];

export function NavBar() {
  const location = useLocation();

  return (
    <header className="border-b border-border bg-background/80 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-4">
        <NavLink to="/" className="flex items-center gap-2 shrink-0">
          <DiamondMark className="h-7 w-7" />
          <span className="text-sm font-semibold tracking-tight text-foreground">
            Steal Decision Model
          </span>
        </NavLink>

        <nav className="hidden md:block">
          <AnimatedBackground
            defaultValue={location.pathname}
            enableHover={false}
            className="rounded-md bg-secondary"
            transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
          >
            {NAV_ITEMS.map((item) => {
              const isActive =
                item.to === '/'
                  ? location.pathname === '/'
                  : location.pathname.startsWith(item.to);
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  data-id={item.to}
                  end={item.to === '/'}
                  className={cn(
                    'relative z-10 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                    isActive
                      ? 'text-foreground'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  {item.label}
                </NavLink>
              );
            })}
          </AnimatedBackground>
        </nav>

        {location.pathname !== '/predictor' && (
          <Button asChild size="sm">
            <NavLink to="/predictor">Try it</NavLink>
          </Button>
        )}
      </div>
    </header>
  );
}
