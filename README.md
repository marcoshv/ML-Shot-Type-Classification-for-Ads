# Shot Type Classification for Ads
> Shot Type Classification for Ads

## 1. Install

You can use `Docker` to install all the needed packages and libraries easily. Two Dockerfiles are provided for both CPU and GPU support.

- **CPU:**

```bash
$ docker build -t sp_final_mhv -f docker/Dockerfile .
```
```bash
$ docker build -t sp_final_mhv --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile .
```


- **GPU:**

```bash
$ docker build -t sp_final_mhv -f docker/Dockerfile_gpu .
```
```bash
$ docker build -t sp_final_mhv --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile_gpu .
```

### Run Docker

- **CPU:**

```bash
$ docker run --rm --net host -it -v "$(pwd):/home/app/src" --workdir /home/app/src sp_final_mhv bash
```

- **GPU:**

```bash
$ docker run --rm --net host --gpus all -it -v "$(pwd):/home/app/src" --workdir /home/app/src sp_final_mhv bash
```
