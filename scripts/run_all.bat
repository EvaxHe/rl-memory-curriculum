@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Full experiment pipeline for rl-memory-curriculum (Windows)
REM
REM Trains 3 curriculum configs (AA + MM), evaluates on 2 benchmarks,
REM runs analysis and generates paper tables.
REM
REM Usage:
REM   scripts\run_all.bat                           (default: LoRA, >=48GB GPU)
REM   scripts\run_all.bat --config-dir configs\full_ft
REM   scripts\run_all.bat --dry-run                (quick eval/analysis)
REM ============================================================

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv is not installed.
    echo Install uv: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

set "CONFIG_DIR=configs"
set "DRY_RUN=false"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=true"
    shift
    goto parse_args
)
if /I "%~1"=="--config-dir" (
    if "%~2"=="" (
        echo ERROR: --config-dir requires a value.
        exit /b 1
    )
    set "CONFIG_DIR=%~2"
    shift
    shift
    goto parse_args
)
set "ARG=%~1"
if /I "!ARG:~0,13!"=="--config-dir=" (
    set "CONFIG_DIR=!ARG:~13!"
    shift
    goto parse_args
)
shift
goto parse_args

:args_done
set "CONFIG_A=%CONFIG_DIR%\train_locomo_only.yaml"
set "CONFIG_B=%CONFIG_DIR%\train_mixed.yaml"
set "CONFIG_C=%CONFIG_DIR%\train_longmemeval_only.yaml"
set "EVAL_CONFIG=configs\eval.yaml"
set "RESULTS_DIR=results"
set "LONGMEMEVAL_DIR=data\raw\longmemeval\data"
set "LONGMEMEVAL_CLEAN=%LONGMEMEVAL_DIR%\longmemeval_s_cleaned.json"
set "LONGMEMEVAL_S=%LONGMEMEVAL_DIR%\longmemeval_s.json"
set "LONGMEMEVAL_URL=https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

echo ============================================
echo   rl-memory-curriculum experiment
echo ============================================
echo Config dir: %CONFIG_DIR%
echo Dry run:    %DRY_RUN%
echo Start:      %date% %time%
echo.

for %%F in ("%CONFIG_A%" "%CONFIG_B%" "%CONFIG_C%") do (
    if not exist "%%~F" (
        echo ERROR: Config not found: %%~F
        exit /b 1
    )
)

REM Ensure LongMemEval raw file is present. Clone alone does not include this file.
if not exist "%LONGMEMEVAL_CLEAN%" if not exist "%LONGMEMEVAL_S%" (
    echo LongMemEval raw json not found. Auto-downloading to %LONGMEMEVAL_CLEAN%
    if not exist "%LONGMEMEVAL_DIR%" mkdir "%LONGMEMEVAL_DIR%"
    curl -L --fail -o "%LONGMEMEVAL_CLEAN%" "%LONGMEMEVAL_URL%"
    if errorlevel 1 (
        echo WARN: curl failed, trying PowerShell download...
        powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri '%LONGMEMEVAL_URL%' -OutFile '%LONGMEMEVAL_CLEAN%'"
        if errorlevel 1 (
            echo ERROR: Failed to download LongMemEval data file.
            echo Download manually: %LONGMEMEVAL_URL%
            exit /b 1
        )
    )
)

echo === Step 0: GPU Check ===
uv run python -c "import torch; assert torch.cuda.is_available(), 'No GPU found.'; name=torch.cuda.get_device_name(0); vram=torch.cuda.get_device_properties(0).total_memory/1e9; print(f'GPU: {name} ({vram:.1f} GB)')"
if errorlevel 1 goto :error
echo.

set "EVAL_EXTRA="
if /I "%DRY_RUN%"=="true" (
    echo DRY RUN MODE: limited eval (5 examples per benchmark^)
    echo Training is NOT shortened - dry-run only affects eval + analysis
    set "EVAL_EXTRA=--max-examples 5"
    echo.
)

echo === Step 1: Data Preparation ===
uv run python data\prepare_locomo.py
if errorlevel 1 goto :error
uv run python data\prepare_longmemeval.py
if errorlevel 1 goto :error
uv run python data\prepare_mixed.py
if errorlevel 1 goto :error

REM Data prep scripts can print errors and still exit 0; verify outputs explicitly.
if not exist "data\processed\locomo_train.jsonl" (
    echo ERROR: Missing data\processed\locomo_train.jsonl after Step 1.
    goto :error
)
if not exist "data\processed\longmemeval_train.jsonl" (
    echo ERROR: Missing data\processed\longmemeval_train.jsonl after Step 1.
    goto :error
)
if not exist "data\processed\mixed_train.jsonl" (
    echo ERROR: Missing data\processed\mixed_train.jsonl after Step 1.
    goto :error
)
echo.

for /f %%T in ('powershell -NoProfile -Command "[int][double]::Parse(((Get-Date).ToUniversalTime() - [datetime]''1970-01-01'').TotalSeconds.ToString())"') do set "START_TS=%%T"

call :train_one A "%CONFIG_A%" config_a_locomo_only answer_agent "Answer Agent"
if errorlevel 1 goto :error
call :train_one B "%CONFIG_B%" config_b_mixed answer_agent "Answer Agent"
if errorlevel 1 goto :error
call :train_one C "%CONFIG_C%" config_c_longmemeval_only answer_agent "Answer Agent"
if errorlevel 1 goto :error

call :train_one A "%CONFIG_A%" config_a_locomo_only memory_manager "Memory Manager"
if errorlevel 1 goto :error
call :train_one B "%CONFIG_B%" config_b_mixed memory_manager "Memory Manager"
if errorlevel 1 goto :error
call :train_one C "%CONFIG_C%" config_c_longmemeval_only memory_manager "Memory Manager"
if errorlevel 1 goto :error

echo === Step 8: Evaluation ===
echo Start: %date% %time%
uv run python -m src.eval.runner --config "%EVAL_CONFIG%" --skip-judge %EVAL_EXTRA%
if errorlevel 1 goto :error
echo Eval done: %date% %time%
echo.

echo === Step 9: Analysis ===
uv run python -m src.eval.analyze --results "%RESULTS_DIR%\all_results.json" --output paper\tables\
if errorlevel 1 goto :error
echo.

for /f %%T in ('powershell -NoProfile -Command "[int][double]::Parse(((Get-Date).ToUniversalTime() - [datetime]''1970-01-01'').TotalSeconds.ToString())"') do set "END_TS=%%T"
set /a ELAPSED_MIN=(END_TS-START_TS)/60
set /a HOURS=ELAPSED_MIN/60
set /a MINS=ELAPSED_MIN%%60

echo ============================================
echo   PIPELINE COMPLETE
echo ============================================
echo Total time: %HOURS%h %MINS%m
echo End:        %date% %time%
echo Results:    %RESULTS_DIR%\
echo Tables:     paper\tables\
echo ============================================
exit /b 0

:error
echo.
echo ERROR: Pipeline failed.
exit /b 1

:train_one
set "T_LABEL=%~1"
set "T_CONFIG=%~2"
set "T_PREFIX=%~3"
set "T_AGENT=%~4"
set "T_TITLE=%~5"
set "T_CKPT=checkpoints\%T_PREFIX%\%T_AGENT%"

if exist "%T_CKPT%\NUL" goto :train_skip

echo === Training Config %T_LABEL% - %T_TITLE% ===
echo Config: %T_CONFIG%
echo Start: %date% %time%
uv run python -m src.train.grpo --config "%T_CONFIG%" --agent %T_AGENT%
if errorlevel 1 exit /b 1

if /I "%T_AGENT%"=="answer_agent" goto :train_done_aa
echo Config %T_LABEL% MM done: %date% %time%
echo.
exit /b 0

:train_done_aa
echo Config %T_LABEL% AA done: %date% %time%
echo.
exit /b 0

:train_skip
if /I "%T_AGENT%"=="answer_agent" goto :train_skip_aa
echo === Config %T_LABEL% MM - SKIPPED (checkpoint exists) ===
exit /b 0

:train_skip_aa
echo === Config %T_LABEL% AA - SKIPPED (checkpoint exists) ===

exit /b 0
