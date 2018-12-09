IMAGE_NAME := proto-speaker-recognition

.PHONY: help
help:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: image
image:	## Build docker image
	docker build -t $(IMAGE_NAME) .

.PHONY: run
run:	## Run docker image using default options
	docker run -ti $(IMAGE_NAME)


