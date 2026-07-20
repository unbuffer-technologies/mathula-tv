import ast,json
from pathlib import Path

import mathula_tv


def test_version_increment_and_executable_dedicated_notebook():
    assert mathula_tv.__version__=="0.1.3"
    path=Path(__file__).parents[1]/"notebooks/mathula_tv_omnivoice_worker.ipynb"; notebook=json.loads(path.read_text())
    cells=[c for c in notebook["cells"] if c["cell_type"]=="code"]
    assert len(cells)==18
    code="\n".join("".join(c["source"]) for c in cells)
    assert "omnivoice==0.2.1" in code and "pyannote.audio" not in code.lower() and "run_pyannote" not in code and "--force-reinstall" in code
    for i,cell in enumerate(cells):
        source="\n".join(line for line in "".join(cell["source"]).splitlines() if not line.lstrip().startswith("%"))
        ast.parse(source,filename=f"cell-{i+1}")
