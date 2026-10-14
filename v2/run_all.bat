@echo off
rem Runs the whole v2 pipeline end to end from the project root.
rem Every step skips work whose outputs already exist; pass --force to a step to redo it.
cd /d "%~dp0.."
set PY=fault-env\Scripts\python.exe
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_XET=1

%PY% v2\build_dataset.py || goto :fail
%PY% v2\train_transformer.py --model distilbert-base-uncased --name distilbert || goto :fail
%PY% v2\train_transformer.py --model FacebookAI/roberta-base --name roberta || goto :fail
rem ELECTRA weights live in models\pretrained (downloaded with huggingface_hub snapshot_download, local_dir=...)
%PY% v2\train_transformer.py --model models/pretrained/electra-base-discriminator --name electra || goto :fail
%PY% v2\train_classical.py --name hgb || goto :fail
%PY% v2\evaluate.py || goto :fail
%PY% v2\select_model.py || goto :fail
%PY% v2\evaluate.py --ablation || goto :fail
%PY% v2\evaluate.py --report || goto :fail
rem needs Ollama running for real LLM cards (otherwise cards fall back); writes results\model_report.md
%PY% v2\compare_demo.py || goto :fail

echo.
echo Pipeline finished. Now, in two terminals:
echo   %PY% v2\daemon.py --demo
echo   fault-env\Scripts\streamlit.exe run v2\dashboard.py   (any of the 3 models can be started from its sidebar)
echo then re-run "%PY% v2\evaluate.py --report" to add the LLM latency to results\RUN_REPORT.md
exit /b 0

:fail
echo Step failed with exit code %errorlevel%.
exit /b %errorlevel%
