@echo off
chcp 65001 >nul
cd /d "C:\Users\gbenn\OneDrive\Documents\Claude\Projects\crypto project"
"C:\Users\gbenn\AppData\Local\Python\bin\python.exe" -X utf8 crypto_predictor.py --output "C:\Users\gbenn\OneDrive\Documents\Claude\Projects\crypto project" > task_log.txt 2>&1
