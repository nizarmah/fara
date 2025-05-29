.PHONY: build run

build:
	go mod download
	go build -o fara cmd/fara/main.go

run: build
	./fara
