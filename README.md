```
docker build . -t ishiki-api
docker run -it \
    -v /Users/musui/ishiki/ishiki-api/api:/app/api \
    -v /Users/musui/ishiki/ishiki-api/tests:/app/tests \
    -p 5000:5000 \
    ishiki-api /bin/bash
```

```
export FLASK_APP=api/__init__.py && export FLASK_ENV=development && flask run --host 0.0.0.0
waitress-serve --port 5000 api:app
```

```
pytest -o log_cli=true
```