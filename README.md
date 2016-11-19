# mesen
mesen uses Google Cloud Vision API to detect face landmarks.

```sh
export GOOGLE_APPLICATION_CREDENTIALS=credentials.json
python main.py
```

```sh
curl -sL -X POST localhost:8080 -F 'image=@data/uesaka.jpg' | imgcat
```

## Run app on docker
It uses docker to deploy to google app engine.

You can launch the docker environment by following commands:
```sh
docker build -t mesen .
docker run -p 8080:8080 mesen
```

and open `http:://$(docker-machine ip):8080`

## Deploy
It's running on a container with flexible enviroment in google app engine 

```sh
gcloud app deploy
```

