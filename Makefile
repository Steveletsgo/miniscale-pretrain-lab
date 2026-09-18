.PHONY: test smoke

test:
	PYTHONPATH=src python -m pytest -q

smoke:
	PYTHONPATH=src python -m miniscale.train --config configs/smoke_cpu.yaml

