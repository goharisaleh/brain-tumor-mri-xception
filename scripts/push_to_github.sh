#!/usr/bin/env bash
# First upload of this project to GitHub.
# Run it from inside the project folder:   bash scripts/push_to_github.sh
set -euo pipefail

GH_USER="goharisaleh"
REPO="brain-tumor-mri-xception"
DESC="Brain tumor MRI classification (glioma, meningioma, pituitary, normal) with Xception transfer learning"

git init -q 2>/dev/null || true
git add .
git commit -q -m "Initial commit: brain tumor MRI classification with Xception" || echo "(nothing new to commit)"
git branch -M main

if command -v gh >/dev/null 2>&1; then
  # GitHub CLI: creates the repository and pushes in one step
  gh auth status >/dev/null 2>&1 || gh auth login
  gh repo create "$GH_USER/$REPO" --public --description "$DESC" --source=. --remote=origin --push
  gh repo edit "$GH_USER/$REPO" --add-topic deep-learning,medical-imaging,brain-tumor,mri,transfer-learning,ensemble-learning,tensorflow,keras || true
else
  # Plain git: first create an EMPTY repo named $REPO on https://github.com/new (no README, no license)
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/$GH_USER/$REPO.git"
  echo "When asked for a password, paste a GitHub Personal Access Token (Settings -> Developer settings -> Tokens)."
  git push -u origin main
fi
echo "Done: https://github.com/$GH_USER/$REPO"
