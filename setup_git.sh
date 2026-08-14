#!/bin/bash
# Run this ONCE to initialize the repo and make your first commit

echo "Setting up Git repository..."

git init

git add .

git commit -m "feat: initial project structure and dataset download

- Add full folder structure (data, notebooks, src, app, models, reports, docs)
- Add README with project overview and 14-day progress tracker
- Add requirements.txt with all NLP dependencies
- Add .gitignore for Python/ML project
- Add docs/data_download.md with AfriSenti dataset instructions
- Add docs/daily_commit_guide.md with 14-day commit plan"

echo ""
echo "✓ Git initialized with first commit."
echo ""
echo "Next steps:"
echo "  1. Create repo on GitHub: github.com/new"
echo "     Name: amharic-sentiment-analysis"
echo "  2. Run:"
echo "     git remote add origin https://github.com/YOUR_USERNAME/amharic-sentiment-analysis.git"
echo "     git branch -M main"
echo "     git push -u origin main"
