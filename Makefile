.PHONY: build run

build:
	go build -o fara cmd/fara/main.go

run: build
	./fara
