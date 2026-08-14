# GitHub 上传步骤

## 1. 在 GitHub 创建仓库

访问 https://github.com/new 创建新仓库：
- Repository name: math-modeling-competition
- Description: 数学建模竞赛核心算法库 - CUMCM/GMC 专用
- Visibility: Private
- 不要初始化 README 或 .gitignore（已存在）
- 点击 "Create repository"

## 2. 添加远程仓库并推送

```bash
cd "D:\本地的知识库构建\math-modeling-competition"
git remote add origin https://github.com/lzc18/math-modeling-competition.git
git branch -M main
git push -u origin main
```

## 3. 或者使用 gh CLI（如果已安装）

```bash
gh repo create math-modeling-competition --private --source=. --push
```
