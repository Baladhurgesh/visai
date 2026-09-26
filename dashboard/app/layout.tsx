import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "Visai", description: "Self-improving model & kernel optimization agent" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
