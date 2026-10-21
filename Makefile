export PYTHONPATH := .:src
setup:        ## install deps and fetch the corpus
	pip install -r requirements-dev.txt && python scripts/fetch_corpus.py
test:
	pytest -q
validate:     ## check gold evidence exists at every chunk size
	python eval/validate_questions.py
ablate:       ## real models (BGE + bge-reranker); writes results/ablation_real.md
	python eval/run_ablations.py
refusal:
	python eval/refusal_eval.py
eval:         ## needs RAG_OPENAI_BASE_URL, RAG_OPENAI_API_KEY and MODEL=<model name>; writes results/e2e.json
	python -m eval.run_e2e --tag e2e --generator openai:$(MODEL) --judge openai:$(MODEL)
offline:      ## no model downloads, no API key: pipeline smoke test with stand-ins
	python eval/run_ablations.py --offline && python -m eval.run_e2e --offline --tag offline_e2e
serve:
	uvicorn ragkit.api:app --port 8000
ui:
	streamlit run app/streamlit_app.py
