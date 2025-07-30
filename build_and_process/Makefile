# Makefile for Vietnamese Sign Language Video Scraper

IMAGE_NAME = vsl-scraper
CONTAINER_NAME = vsl-scraper-container
APP_DIR = build_and_process
APP_MOUNT = $(APP_DIR)/app
REQUIREMENTS = $(APP_DIR)/requirements.txt

.PHONY: help build run test clean

help:
	@echo "Available targets:"
	@echo "  make build    - Build the Docker image"
	@echo "  make run      - Run scraper inside Docker"
	@echo "  make test     - Run tests inside Docker"
	@echo "  make clean    - Remove JSON output and image"
	@echo "  make github   - Test GitHub Actions locally with act (optional)"

build:
	docker build -t $(IMAGE_NAME) $(APP_DIR)

run:
	docker run --rm -v $(PWD)/$(APP_MOUNT):/app/app $(IMAGE_NAME)

test:
	docker run --rm -v $(PWD)/$(APP_DIR):/app -w /app $(IMAGE_NAME) pytest test/

clean:
	rm -f $(APP_MOUNT)/vietnamese_sign_language_videos.json
	docker rmi -f $(IMAGE_NAME) || true

github:
	@echo "Running GitHub Actions locally via act (requires installation)"
	act

all: build run test

