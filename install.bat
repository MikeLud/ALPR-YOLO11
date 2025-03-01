:: Installation script :::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::
::                           ALPR (YOLOv8)
::
:: This script is only called from ..\..\CodeProject.AI-Server\src\setup.bat in
:: Dev setup, or ..\..\src\setup.bat in production

@if "%1" NEQ "install" (
    echo This script is only called from ..\..\CodeProject.AI-Server\src\setup.bat
    @pause
    @goto:eof
)

:: Create directories if they don't exist
if not exist "models" mkdir "models"
if not exist "test" mkdir "test"
