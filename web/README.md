# Nigeria Treasury Bill Rate Predictor — Static Web App

A static browser app for estimating the next Nigerian Treasury Bill stop rate
for the 91-day, 182-day, and 364-day tenors.

Open `index.html` in a browser. No build step or server is required.

## What It Uses

- Latest auction baseline from the 20 May 2026 NTB auction.
- Recent auction trend across the three standard tenors.
- User assumptions for MPR, inflation, offer size, expected subscriptions,
  system liquidity, and naira pressure.

## Live API Integration

If the FastAPI backend (`backend_main.py`) is running, the app will
automatically call `/snapshots/latest` on load and:

- Append the most recent market snapshot to the auction table.
- Pre-fill MPR, inflation, and system liquidity from the live data.

If the backend is unreachable the app falls back silently to the hardcoded
seed data — no error is shown to the user.

## Model Notes

The forecast is an indicative scenario model, not a statistical investment
model. It starts from the latest stop rate and applies bounded adjustments for:

- Recent tenor-specific trend (mean of consecutive auction-to-auction moves).
- Bid-cover ratio.
- MPR-to-stop-rate gap.
- Inflation pressure.
- System liquidity.
- FX / naira pressure.
- Tenor premium.

The confidence label reflects how far the predicted rate deviates from the
baseline: small adjustments yield "High confidence"; large adjustments (the
model is extrapolating further from recent pricing) yield lower confidence.

The output is useful for scenario thinking and explaining likely direction, but
official auction circulars and professional market data should remain the source
of truth for trading or investment decisions.
