@echo off
REM Corrida larga 2 000 000 pasos: inicializaciones 2, 3 y 4 en este PC.
REM Si se corta (corte de luz, reinicio, error), vuelve a lanzarla: sigue desde el ultimo punto de control.
cd /d %~dp0
call .venv\Scripts\activate.bat
:otra
python corrida_larga.py --base resultados\completa --out resultados\larga_2M --inits 2,3,4
if errorlevel 1 (
  echo La corrida se corto. Reintentando en 60 s...
  timeout /t 60 /nobreak
  goto otra
)
echo LISTO. Copia la carpeta resultados\larga_2M a la nube.
pause
