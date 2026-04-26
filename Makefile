.PHONY: feature-store modeling research validate test test-notebooks

feature-store:
	python scripts/run_feature_store.py

modeling:
	python scripts/run_multi_horizon_modeling.py

research:
	python scripts/run_research_pipeline.py

validate:
	python scripts/validate_research_outputs.py

test:
	python -m unittest discover -s tests -v

test-notebooks:
	RUN_NOTEBOOK_INTEGRATION=1 python -m unittest tests.test_notebook_execution -v
