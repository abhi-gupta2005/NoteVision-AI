# vscode-config/

Ready-made VS Code settings for this project (interpreter path, pytest, launch
configurations for the Streamlit app and the tests).

`scripts\setup.ps1` / `scripts/setup.sh` copy these files into `.vscode\` automatically
when that folder does not exist yet. To do it by hand:

```powershell
Copy-Item -Recurse vscode-config .vscode
```

Then open the folder in VS Code, select the `.venv` interpreter if it is not picked up
automatically (Ctrl+Shift+P -> "Python: Select Interpreter"), and press F5 to start the app.
