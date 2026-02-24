#!/usr/bin/env bash
set -euo pipefail

echo "📥 Pull (rebase)..."
git pull --rebase origin main

echo "🧪 Running tests..."
./pytest.sh -q

echo "➕ Staging tracked changes only..."
git add -u

echo "➕ Staging new files (except ignored)..."
git add .

if git diff --cached --quiet; then
  echo "✅ No changes to commit."
  exit 0
fi

echo "💾 Committing..."
git commit -m "Auto sync: code updates"

echo "🚀 Pushing..."
git push origin main

echo "✅ Done."
