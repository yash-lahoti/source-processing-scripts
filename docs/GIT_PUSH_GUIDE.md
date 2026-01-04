# Git Push Guide: Ensuring Code is Pushed to origin/main

## Quick Check Commands

### 1. Check Current Status
```bash
git status
```

Should show:
- `On branch main`
- `Your branch is up to date with 'origin/main'`
- `nothing to commit, working tree clean`

### 2. Verify Remote is Set
```bash
git remote -v
```

Should show:
```
origin  git@github.com:yash-lahoti/source-processing-scripts.git (fetch)
origin  git@github.com:yash-lahoti/source-processing-scripts.git (push)
```

### 3. Check Branch Tracking
```bash
git branch -vv
```

Should show:
```
* main  f7af1e7 [origin/main] run-script
```

The `[origin/main]` indicates your local branch is tracking the remote.

## Pushing to origin/main

### Step-by-Step Process

1. **Stage all changes:**
   ```bash
   git add .
   ```
   Or stage specific files:
   ```bash
   git add file1.py file2.py
   ```

2. **Commit changes:**
   ```bash
   git commit -m "Your commit message"
   ```

3. **Push to origin/main:**
   ```bash
   git push -u origin main
   ```
   
   The `-u` flag sets up tracking (only needed the first time). After that, you can just use:
   ```bash
   git push
   ```

### Verify Push Was Successful

After pushing, run:
```bash
git status
```

Should show:
```
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

## Common Scenarios

### Scenario 1: New Files Not Pushed

If you have new files that aren't tracked:

```bash
# Check what's not tracked
git status

# Add new files
git add new_file.py

# Commit
git commit -m "Add new_file.py"

# Push
git push origin main
```

### Scenario 2: Modified Files Not Pushed

If you modified existing files:

```bash
# See what changed
git status

# Stage changes
git add modified_file.py

# Commit
git commit -m "Update modified_file.py"

# Push
git push origin main
```

### Scenario 3: Already Up to Date

If `git status` shows "Your branch is up to date with 'origin/main'", then everything is already pushed!

## Force Push (Use with Caution!)

⚠️ **Only use if you're sure!** This overwrites remote history.

```bash
git push --force origin main
```

Or safer:
```bash
git push --force-with-lease origin main
```

## Troubleshooting

### "Your branch is ahead of 'origin/main'"

This means you have local commits not pushed yet:

```bash
git push origin main
```

### "Your branch and 'origin/main' have diverged"

This means both local and remote have different commits:

```bash
# Pull first, then push
git pull origin main
git push origin main
```

### "Remote repository not found"

The remote URL might be wrong or the repository doesn't exist:

```bash
# Check remote
git remote -v

# Update remote if needed
git remote set-url origin git@github.com:yash-lahoti/source-processing-scripts.git
```

## Quick Reference

```bash
# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Your message"

# Push to origin/main
git push origin main

# Or just (if tracking is set up)
git push
```

## Your Current Setup

- **Remote:** `git@github.com:yash-lahoti/source-processing-scripts.git`
- **Branch:** `main`
- **Tracking:** `origin/main` ✅

Everything is properly configured!

