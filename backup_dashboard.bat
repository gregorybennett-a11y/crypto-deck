@echo off
chcp 65001 >nul
set PROJ=C:\Users\gbenn\OneDrive\Documents\Claude\Projects\crypto project
set PYTHON=C:\Users\gbenn\AppData\Local\Python\bin\python.exe

:: Use Python to create the dated backup folder and copy files directly
"%PYTHON%" -c ^
"import shutil, os, datetime; ^
proj=r'%PROJ%'; ^
dst=os.path.join(proj,'backups',str(datetime.date.today())); ^
os.makedirs(dst,exist_ok=True); ^
[shutil.copy2(os.path.join(proj,f),dst) for f in ['crypto_dashboard.html','crypto_predictor.py','run_dashboard.bat'] if os.path.exists(os.path.join(proj,f))]; ^
print('Backup saved to '+dst)"
