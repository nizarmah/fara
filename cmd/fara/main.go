// Command fara helps control the mouse via the keyboard.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	hook "github.com/robotn/gohook"
)

var (
	quitHotkey = []string{"ctrl", "shift", "q"}
)

func main() {
	// Context.
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(errors.New("shutdown"))

	// Handle OS signals.
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGKILL)
	go func() { cancel(fmt.Errorf("signal: %v", <-signalChan)) }()

	// Register key events.
	registerKeyEvents(ctx, cancel)

	// Start the hook.
	events := hook.Start()
	defer hook.End()
	go hook.Process(events)

	log.Println("Fara is running...")
	log.Println(
		fmt.Sprintf(
			"Press %q to quit.",
			strings.Join(quitHotkey, "+"),
		),
	)

	// Wait for the context to be done.
	<-ctx.Done()

	log.Println("Fara is quitting...")
	log.Println(fmt.Sprintf("Reason: %v", context.Cause(ctx)))
}

// registerKeyEvents registers key events.
func registerKeyEvents(
	_ context.Context,
	cancel context.CancelCauseFunc,
) {
	// Add a hook to quit the program.
	hook.Register(hook.KeyDown, quitHotkey, func(_ hook.Event) {
		cancel(errors.New("quit hotkey"))
	})

	// Todo: Add other key events here.
}
