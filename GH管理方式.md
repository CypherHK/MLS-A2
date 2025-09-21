你完全可以在GitHub上创建多个分支来保存不同版本的项目，例如 `main` 和 `advanced` 分支。这样，你可以将不同的版本推送到不同的分支，并保留它们。以下是操作步骤：

### 1. **初始化本地Git仓库（如果尚未初始化）**

如果你还没有初始化本地Git仓库，可以先进行初始化：

```bash
cd /path/to/your/project  # 进入你的项目文件夹
git init  # 初始化Git仓库
```

### 2. **将本地项目添加到GitHub远程仓库**

如果你已经创建了一个GitHub仓库，接下来将本地项目推送到GitHub。

* 在GitHub上创建一个新的仓库（例如 `my-project`）。
* 在本地项目目录下，将GitHub仓库链接添加为远程仓库：

```bash
git remote add origin https://github.com/yourusername/my-project.git
```

### 3. **创建 `main` 分支并推送到GitHub**

如果你要将当前版本推送到 `main` 分支：

```bash
git add .  # 将所有更改添加到暂存区
git commit -m "Initial commit"  # 提交更改
git branch -M main  # 确保当前分支是main，如果不是，可以重命名为main
git push -u origin main  # 将main分支推送到GitHub
```

### 4. **创建 `advanced` 分支**

假设你已经在 `main` 分支上推送了第一个版本。现在，你可以创建一个 `advanced` 分支，并在该分支上进行修改：

```bash
git checkout -b advanced  # 创建并切换到advanced分支
```

现在，你可以在 `advanced` 分支上进行开发、添加新功能或做其他修改。

### 5. **推送 `advanced` 分支到GitHub**

你完成了 `advanced` 分支的修改后，可以将它推送到GitHub：

```bash
git add .  # 添加修改到暂存区
git commit -m "Add advanced version"  # 提交修改
git push -u origin advanced  # 将advanced分支推送到GitHub
```

### 6. **在GitHub上查看和管理分支**

现在，你的GitHub仓库中就有两个分支：`main` 和 `advanced`。你可以在GitHub的网页界面上切换不同的分支，查看每个分支的内容。

* 进入你的GitHub仓库页面。
* 在右上角的“Branch”下拉菜单中选择切换到不同的分支（例如 `main` 或 `advanced`）。

### 7. **未来的更新**

* 每次修改后，你都可以切换到相应的分支，进行修改并推送。

  * 对于 `main` 分支：

    ```bash
    git checkout main
    git add .  # 添加修改
    git commit -m "Update main version"
    git push origin main
    ```
  * 对于 `advanced` 分支：

    ```bash
    git checkout advanced
    git add .  # 添加修改
    git commit -m "Update advanced version"
    git push origin advanced
    ```

### 总结

* 你可以创建多个分支，例如 `main` 和 `advanced`，分别保存不同版本的代码。
* 每个分支可以独立进行开发，推送到GitHub时不会覆盖其他分支的内容。
* 在GitHub界面上，你可以随时切换分支查看不同版本的代码。

通过这种方式，你就能在GitHub上有效管理不同版本的项目了。
