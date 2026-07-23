import { Outlet } from 'react-router';
import { NavBar } from '@/components/layout/NavBar';
import { Footer } from '@/components/layout/Footer';

export function AppLayout() {
  return (
    <div className="flex min-h-svh flex-col">
      <NavBar />
      <main className="flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
