# Git & GitHub Workflow Guide

Complete guide to version control and shipping your RAG project to GitHub.

---

## What You're Going to Do

1. **Initialize Git** in your project folder
2. **Stage your files** (tell Git what to track)
3. **Commit** (save a snapshot of your work)
4. **Create GitHub repository** (cloud storage)
5. **Push** (upload your code to GitHub)

---

## Prerequisites

### Check if Git is installed

```bash
git --version
```

If not installed:
- **Ubuntu/Debian:** `sudo apt-get install git`
- **macOS:** `brew install git` or install Xcode Command Line Tools
- **Windows:** Download from https://git-scm.com/

### Configure Git (First-Time Setup)

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

This stamps your commits with your identity.

---

## Step 1: Organize Your Project

Before initializing Git, make sure your project is structured properly.

### Copy Files to Project Root

Assuming you have this structure:

```bash
# Your current structure (example)
/home/tanmay/Distroboxes/urban-planning-rag/
├── src/
│   ├── check_docs.py
│   ├── embed_lightning.py
│   ├── retrieval-test.py
│   └── test_gemini.py
├── docs/
│   ├── swm_2016.pdf
│   ├── urdpfi_vol1.pdf
│   └── urdpfi_vol2.pdf
├── data/
│   ├── embeddings/
│   │   ├── embeddings.pt
│   │   └── metadata.json
│   └── page_images/
└── (your notebooks from Lightning.ai)
```

### Create Proper Structure

```bash
cd /home/tanmay/Distroboxes/urban-planning-rag

# Create directories
mkdir -p notebooks scripts

# Move notebooks (if you have them locally)
mv embed_docs.ipynb notebooks/
mv rag.ipynb notebooks/

# Move utility scripts
mv src/check_docs.py scripts/
mv src/test_gemini.py scripts/

# The new src/rag.py you created should be in src/
# (from the files I just generated for you)
```

---

## Step 2: Initialize Git Repository

```bash
cd /home/tanmay/Distroboxes/urban-planning-rag

# Initialize Git
git init
```

You'll see:
```
Initialized empty Git repository in /home/tanmay/Distroboxes/urban-planning-rag/.git/
```

This creates a hidden `.git` folder that tracks all your changes.

---

## Step 3: Add Files to Git

### Check Current Status

```bash
git status
```

You'll see a list of "untracked files" (files Git doesn't know about yet).

### Stage Files

**Option A: Stage everything (recommended for first commit)**
```bash
git add .
```

**Option B: Stage specific files**
```bash
git add README.md
git add requirements.txt
git add src/
git add cli.py
```

### Verify What's Staged

```bash
git status
```

You should see files listed under "Changes to be committed" in green.

---

## Step 4: Create .gitignore (Important!)

Before committing, create `.gitignore` to exclude large files:

```bash
# The .gitignore file I created earlier should already be in your project root
# Verify it exists:
cat .gitignore
```

This tells Git to IGNORE:
- `data/embeddings/embeddings.pt` (too large for GitHub)
- `data/page_images/` (too many files)
- `.env` (API keys - sensitive!)
- Virtual environments

### Verify Ignored Files

```bash
git status --ignored
```

You should see `embeddings.pt` and `page_images/` listed under "Ignored files".

---

## Step 5: Make Your First Commit

A commit is a snapshot of your project at a specific point in time.

```bash
git commit -m "Initial commit - Urban Planning RAG MVP"
```

**Commit message guidelines:**
- Present tense: "Add feature" not "Added feature"
- Be concise but descriptive
- First line should be <50 characters

**Example commit messages:**
- `Initial commit - Urban Planning RAG MVP`
- `Add ColQwen embedding pipeline`
- `Fix FAISS indexing bug`
- `Update README with setup instructions`

---

## Step 6: Create GitHub Repository

### Go to GitHub

1. Log in to https://github.com
2. Click the **+** icon (top-right corner)
3. Select **"New repository"**

### Configure Repository

**Repository name:** `urban-planning-rag`  
**Description:** `Visual RAG system for Indian urban planning documents using ColQwen + Gemini`

**Visibility:**
- **Public:** Anyone can see your code (recommended for portfolio)
- **Private:** Only you can see it

**DO NOT** check:
- ❌ Add a README file (you already have one)
- ❌ Add .gitignore (you already have one)
- ❌ Choose a license (you can add this later)

Click **"Create repository"**.

---

## Step 7: Connect Local Repository to GitHub

GitHub will show you a page with setup instructions. Copy the commands under **"…or push an existing repository from the command line"**.

They'll look like this:

```bash
git remote add origin https://github.com/YOUR-USERNAME/urban-planning-rag.git
git branch -M main
git push -u origin main
```

### What These Commands Do

1. **`git remote add origin ...`**  
   Tells Git where to push code (GitHub URL)

2. **`git branch -M main`**  
   Renames your branch to "main" (GitHub's default)

3. **`git push -u origin main`**  
   Uploads your code to GitHub

### Authentication

When you run `git push`, GitHub will ask for credentials:

**Use a Personal Access Token (PAT), NOT your password:**

1. GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name: "urban-planning-rag-laptop"
4. Select scopes: Check **`repo`** (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)

**When prompted for credentials:**
```
Username: your-github-username
Password: paste-your-token-here  # NOT your GitHub password
```

---

## Step 8: Verify Upload

### Check GitHub Website

Go to: `https://github.com/YOUR-USERNAME/urban-planning-rag`

You should see:
- Your README.md displayed
- All your files listed
- Commit history (1 commit)

### Check Local Status

```bash
git status
```

Should say: `On branch main` with no uncommitted changes.

---

## Daily Git Workflow

### After Making Changes

```bash
# 1. Check what changed
git status

# 2. Stage changes
git add .  # Or specify files: git add src/rag.py

# 3. Commit with message
git commit -m "Add query caching feature"

# 4. Push to GitHub
git push
```

### View Commit History

```bash
git log --oneline
```

### Undo Changes (Before Commit)

```bash
# Discard changes to a file
git checkout -- filename.py

# Unstage a file
git reset HEAD filename.py
```

---

## Handling Large Files (embeddings.pt)

GitHub has a **100MB file size limit**. Your `embeddings.pt` is 573MB, so you can't push it directly.

### Option 1: Provide Download Link (Recommended)

1. Upload `embeddings.pt` + `page_images/` to Google Drive
2. Get shareable link
3. Add link to README.md

**Already done in the README I created for you.**

### Option 2: Git LFS (Git Large File Storage)

If you want to version control large files:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "data/page_images/*.png"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Now add the large files
git add data/embeddings/embeddings.pt
git add data/page_images/
git commit -m "Add embeddings and page images"
git push
```

**Note:** GitHub free tier has LFS limits (1GB storage, 1GB bandwidth/month).

---

## Useful Git Commands

### Status & Info

```bash
git status                 # What's changed?
git log --oneline          # Commit history
git diff                   # Show exact changes
git remote -v              # Show GitHub URL
```

### Branching (Advanced)

```bash
git branch feature-name    # Create new branch
git checkout feature-name  # Switch to branch
git checkout main          # Back to main
git merge feature-name     # Merge branch into main
```

### Pulling Changes

If you edit files on GitHub or work from multiple computers:

```bash
git pull                   # Download latest changes
```

---

## Common Issues & Solutions

### "failed to push some refs"

**Cause:** Someone (or you on another machine) pushed changes to GitHub.

**Solution:**
```bash
git pull --rebase
git push
```

### "Authentication failed"

**Cause:** Wrong password or token.

**Solution:** Use Personal Access Token, not GitHub password.

### "Large file detected"

**Cause:** Trying to push file >100MB.

**Solution:**
1. Add to `.gitignore`
2. Remove from Git: `git rm --cached filename`
3. Commit and push

### Accidentally committed large file

```bash
# Remove from last commit
git rm --cached data/embeddings/embeddings.pt
git commit --amend -m "Initial commit (removed large files)"
git push --force
```

---

## Kate Editor Git Integration (Optional)

If you prefer GUI:

1. Open project in Kate
2. **Git → Initialize Repository**
3. **Git → Show Changes** (see modified files)
4. Right-click files → **Stage**
5. **Git → Commit** (write message)
6. **Git → Push**

**Terminal is more reliable**, but Kate's GUI is fine for simple commits.

---

## Next Steps

1. **Write good commit messages** as you develop
2. **Commit frequently** (daily, or after completing features)
3. **Push regularly** to back up your work
4. **Update README** as you add features

---

## Example Workflow Session

```bash
# Morning: Start working
cd ~/Distroboxes/urban-planning-rag
git status  # Check current state

# Make changes to code
nano src/rag.py  # Edit file

# Afternoon: Commit progress
git add src/rag.py
git commit -m "Add caching layer for query embeddings"
git push

# Evening: Add documentation
nano docs/API.md
git add docs/API.md
git commit -m "Document API usage"
git push

# Night: Check history
git log --oneline
```

---

## Resources

- **Git Basics:** https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control
- **GitHub Guides:** https://guides.github.com/
- **Interactive Tutorial:** https://learngitbranching.js.org/

---

## Summary: Your First Push Checklist

- [ ] Project is organized properly
- [ ] `.gitignore` is in place
- [ ] Large files excluded
- [ ] `git init` executed
- [ ] `git add .` to stage files
- [ ] `git commit -m "Initial commit"` to save snapshot
- [ ] GitHub repository created
- [ ] `git remote add origin <URL>` to connect
- [ ] `git push -u origin main` to upload
- [ ] Verify on GitHub website
- [ ] Add Google Drive link for embeddings to README

**You're now shipping code like a professional.** 🚀
