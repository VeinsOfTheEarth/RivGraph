@ECHO OFF

pushd %~dp0

set SOURCEDIR=source
set BUILDDIR=build

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=python -m sphinx
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
