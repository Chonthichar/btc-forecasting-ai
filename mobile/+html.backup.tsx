import type { PropsWithChildren } from "react";

export default function Root({ children }: PropsWithChildren) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />

        <meta
          name="viewport"
          content="width=device-width, initial-scale=1"
        />

        <meta name="theme-color" content="#0b1020" />

        <meta
          name="apple-mobile-web-app-capable"
          content="yes"
        />

        <meta
          name="apple-mobile-web-app-status-bar-style"
          content="black-translucent"
        />

        <meta
          name="apple-mobile-web-app-title"
          content="BTC Forecasting AI"
        />

        <link
          rel="manifest"
          href="/manifest.json"
        />

        <link
          rel="apple-touch-icon"
          href="/logo192.png"
        />

        <title>BTC Forecasting AI</title>
      </head>

      <body>{children}</body>
    </html>
  );
}