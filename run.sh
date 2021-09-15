#!/bin/bash

jupyter notebook --ip=0.0.0.0 --port 8080 --allow-root --NotebookApp.token='' --NotebookApp.password='' &
waitress-serve --port 5000 api:app
