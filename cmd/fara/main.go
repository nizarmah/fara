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
	quitKey = []string{"q", "ctrl", "shift"}
)

func main() {
	// Context.
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(errors.New("program ended"))

	// Handle OS signals.
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGKILL)
	go func() {
		cancel(fmt.Errorf("received signal: %v", <-signalChan))
	}()

	// Register key events.
	registerKeyEvents(ctx, cancel)

	// Start the hook.
	events := hook.Start()
	defer hook.End()

	// Process only registered events.
	go hook.Process(events)

	// Print the quit key.
	log.Println("Fara is running...")
	log.Println(
		fmt.Sprintf(
			"Press %q to quit.",
			strings.Join(quitKey, "+"),
		),
	)

	// Wait for the context to be done.
	<-ctx.Done()
	log.Printf("Fara is quitting: %v", context.Cause(ctx))
}

// registerKeyEvents registers key events.
func registerKeyEvents(
	_ context.Context,
	cancel context.CancelCauseFunc,
) {
	// Add a hook to quit the program.
	hook.Register(hook.KeyDown, quitKey, func(_ hook.Event) {
		cancel(errors.New("quit key pressed"))
	})

	// Todo: Add other key events here.
}
