@echo off
rem Shortcut: run  dub --url "https://youtu.be/..."  from this folder
"%~dp0.venv\Scripts\python.exe" "%~dp0dub.py" %*
