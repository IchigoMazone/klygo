$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX first pass failed.' }
    & bibtex main
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX second pass failed.' }
    & xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX final pass failed.' }
}
finally { Pop-Location }
