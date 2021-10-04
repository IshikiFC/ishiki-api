## 起動方法

```
docker-compose down && docker-compose build && docker-compose up -d
```

以下のサービスが起動する。
* APIサーバー: 5000
* Jupyter: 8080

#### コンテナに入る

```
docker-compose exec api /bin/bash
```

#### ログを見る
```
docker-compose logs -f api
```


#### docker-composeを使わない場合
```
docker build . -t isshki-api
docker run -it \
    -v /Users/musui/ishiki/ishiki-api/:/app \
    ishiki-api /bin/bash
```


## APIサーバ
```
waitress-serve --port 5000 api:app
```

```
export FLASK_APP=api/__init__.py && export FLASK_ENV=development && flask run --host 0.0.0.0
```
flaskのdebugモードはimportエラーが出て動かない。([issue](https://github.com/tensorflow/tensorflow/issues/34607))

## 単体テスト
```
pytest -o log_cli=true
```

## シミュレーション
```
python -m scripts.simulate_game_tamakeri
```