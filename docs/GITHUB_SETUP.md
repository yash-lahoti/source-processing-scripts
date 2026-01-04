# GitHub Repository Setup

## Issue: Repository Not Found

The error "repository not found" means either:
1. The repository doesn't exist on GitHub yet
2. The repository name or username is incorrect
3. You don't have access to the repository

## Solution: Create the Repository First

### Step 1: Create Repository on GitHub

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `patient-aggregator` (or any name you prefer)
3. **Description**: "A configurable Python package to aggregate patient data from multiple Excel files into a single CSV file"
4. **Visibility**: Choose Public or Private
5. **Important**: Do NOT check any boxes (no README, .gitignore, or license)
6. Click **"Create repository"**

### Step 2: Add the Correct Remote

After creating the repository, GitHub will show you the repository URL. Use one of these:

**If your GitHub username is `yash-lahoti`:**
```bash
git remote add origin https://github.com/yash-lahoti/patient-aggregator.git
```

**Or if you prefer SSH:**
```bash
git remote add origin git@github.com:yash-lahoti/patient-aggregator.git
```

### Step 3: Verify Remote
```bash
git remote -v
```

Should show:
```
origin  https://github.com/yash-lahoti/patient-aggregator.git (fetch)
origin  https://github.com/yash-lahoti/patient-aggregator.git (push)
```

### Step 4: Push Your Code
```bash
git push -u origin main
```

You'll be prompted for:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your GitHub password)

### Creating a Personal Access Token

If you need a token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Name it: "Git Push Token"
4. Select scope: `repo` (full control of private repositories)
5. Click "Generate token"
6. Copy the token and use it as your password when pushing

## Alternative: Check Your GitHub Username

If you're not sure of your exact GitHub username:
1. Go to: https://github.com/settings/profile
2. Check your username (it's case-sensitive)
3. Use that exact username in the repository URL

## Quick Commands Summary

```bash
# Remove old remote (if needed)
git remote remove origin

# Add correct remote (replace with your actual username/repo)
git remote add origin https://github.com/YOUR_USERNAME/patient-aggregator.git

# Verify
git remote -v

# Push
git push -u origin main
```

