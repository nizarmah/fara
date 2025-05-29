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

	cursorPkg "github.com/nizarmah/fara/pkg/cursor"
)

const debug = true

const (
	cursorRate  = 60 // Hz
	cursorSpeed = 8  // px
)

var (
	faraHotkey  = []string{"ctrl", "shift", "m"}
	resetHotkey = []string{"ctrl", "shift", "r"}
	quitHotkey  = []string{"ctrl", "shift", "q"}
)

var (
	upKey    = []string{"w"}
	downKey  = []string{"s"}
	leftKey  = []string{"a"}
	rightKey = []string{"d"}
)

var (
	leftClickKey = []string{"space"}
)

func main() {
	// Context.
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(errors.New("shutdown"))

	// Handle OS signals.
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM, syscall.SIGKILL)
	go func() { cancel(fmt.Errorf("signal: %v", <-signalChan)) }()

	// Create a cursor client.
	cursor := cursorPkg.NewClient(cursorPkg.ClientConfig{
		Rate:  cursorRate,
		Speed: cursorSpeed,
	})

	// Start the cursor.
	go cursor.Start(ctx)
	defer cursor.Stop()

	// Register key events.
	registerKeyEvents(ctx, cancel, cursor)

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
	cursor *cursorPkg.Client,
) {
	// Add a hook to quit the program.
	hook.Register(hook.KeyDown, quitHotkey, handleQuit(cancel))

	// Add a hook to toggle the cursor.
	hook.Register(hook.KeyDown, faraHotkey, handleCursorToggle(cursor))

	// Add a hook to reset the cursor.
	hook.Register(hook.KeyDown, resetHotkey, handleCursorReset(cursor))

	// Add a hook to move the cursor up.
	hook.Register(hook.KeyDown, upKey, handleCursorHold(cursor, cursorPkg.DirectionUp, true))
	hook.Register(hook.KeyUp, upKey, handleCursorHold(cursor, cursorPkg.DirectionUp, false))

	// Add a hook to move the cursor down.
	hook.Register(hook.KeyDown, downKey, handleCursorHold(cursor, cursorPkg.DirectionDown, true))
	hook.Register(hook.KeyUp, downKey, handleCursorHold(cursor, cursorPkg.DirectionDown, false))

	// Add a hook to move the cursor left.
	hook.Register(hook.KeyDown, leftKey, handleCursorHold(cursor, cursorPkg.DirectionLeft, true))
	hook.Register(hook.KeyUp, leftKey, handleCursorHold(cursor, cursorPkg.DirectionLeft, false))

	// Add a hook to move the cursor right.
	hook.Register(hook.KeyDown, rightKey, handleCursorHold(cursor, cursorPkg.DirectionRight, true))
	hook.Register(hook.KeyUp, rightKey, handleCursorHold(cursor, cursorPkg.DirectionRight, false))

	// Add a hook to left click.
	hook.Register(hook.KeyDown, leftClickKey, handleCursorClick(cursor, "left"))
}

// handleQuit handles a quit event.
func handleQuit(cancel context.CancelCauseFunc) func(hook.Event) {
	return func(_ hook.Event) {
		cancel(errors.New("quit hotkey"))
	}
}

// handleCursorToggle handles a cursor toggle event.
func handleCursorToggle(cursor *cursorPkg.Client) func(hook.Event) {
	return func(_ hook.Event) {
		cursor.Toggle()
	}
}

// handleCursorReset handles a cursor reset event.
func handleCursorReset(cursor *cursorPkg.Client) func(hook.Event) {
	return func(_ hook.Event) {
		cursor.Reset()
	}
}

// handleCursorHold handles a cursor hold event.
func handleCursorHold(
	cursor *cursorPkg.Client,
	direction cursorPkg.Direction,
	hold bool,
) func(hook.Event) {
	return func(_ hook.Event) {
		cursor.Hold(direction, hold)
	}
}

// handleCursorClick handles a cursor click event.
func handleCursorClick(
	cursor *cursorPkg.Client,
	button string,
) func(hook.Event) {
	return func(_ hook.Event) {
		cursor.Click(button)
	}
}
