
## Dependencies

```bash
pip install torch==2.6.0 flask numpy uvicorn

# NOTE: if you have installed transformers, please skip the following line
pip install -e '../transformers[torch]' --no-build-isolation

uvicorn app:app --host 127.0.0.1 --port 7860 --interface wsgi --log-level debug
# running for the first example may take some time
# visit http://127.0.0.1:7860
```