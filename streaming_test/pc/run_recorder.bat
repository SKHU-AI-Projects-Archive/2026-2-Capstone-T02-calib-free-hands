@echo off
rem Double-click to start the receiver AND the Recorder window (manual start/stop of recording),
rem whatever record_mode says in run_config_manual.yaml. Same as: run_pc.bat --manual
call "%~dp0run_pc.bat" --manual
