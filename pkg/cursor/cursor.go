// Package cursor provides a client to control the cursor.
package cursor

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/go-vgo/robotgo"
)

// ClientConfig is the configuration for the cursor client.
type ClientConfig struct {
	// Rate is the rate at which the cursor is updated (Hz).
	Rate int
	// Speed is the number of pixels the cursor moves per tick.
	Speed int
}

// Client is a client to control the cursor.
type Client struct {
	// Rate is the rate at which the cursor is updated (Hz).
	rate int
	// Speed is the number of pixels the cursor moves per tick.
	speed int

	// active is a flag indicating whether the cursor is active.
	active atomic.Bool
	// up, left, down, right are flags indicating the cursor direction.
	up, left, down, right atomic.Bool

	// cancel cancels the running cursor.
	cancel context.CancelFunc
}

// Direction is a Direction the cursor can move in.
type Direction int

const (
	// DirectionUp is the up direction.
	DirectionUp Direction = iota
	// DirectionDown is the down direction.
	DirectionDown
	// DirectionLeft is the left direction.
	DirectionLeft
	// DirectionRight is the right direction.
	DirectionRight
)

// NewClient creates a new client to control the cursor.
func NewClient(cfg ClientConfig) *Client {
	return &Client{
		rate:   cfg.Rate,
		speed:  cfg.Speed,
		active: atomic.Bool{},
		up:     atomic.Bool{},
		left:   atomic.Bool{},
		down:   atomic.Bool{},
		right:  atomic.Bool{},
		cancel: nil,
	}
}

// Start starts the cursor client.
func (c *Client) Start(parentCtx context.Context) {
	ctx, cancel := context.WithCancel(parentCtx)
	c.cancel = cancel

	go c.startTicker(ctx)
}

// Stop stops the cursor client.
func (c *Client) Stop() {
	c.cancel()
}

// Toggle toggles the cursor active state.
func (c *Client) Toggle() {
	isActive := c.active.Load()
	c.active.Store(!isActive)
}

// IsActive returns the active state of the cursor.
func (c *Client) IsActive() bool {
	return c.active.Load()
}

// Hold holds the cursor in a given direction.
func (c *Client) Hold(dir Direction, state bool) {
	if !c.active.Load() {
		return
	}

	switch dir {
	case DirectionUp:
		c.up.Store(state)
	case DirectionDown:
		c.down.Store(state)
	case DirectionLeft:
		c.left.Store(state)
	case DirectionRight:
		c.right.Store(state)
	}
}

// Click clicks the cursor.
func (c *Client) Click(button string) {
	if !c.active.Load() {
		return
	}

	robotgo.Click(button)
}

// Reset resets the cursor to the center of the screen.
func (c *Client) Reset() {
	if !c.active.Load() {
		return
	}

	boundsX, boundsY, boundsWidth, boundsHeight := robotgo.GetDisplayBounds(robotgo.DisplayID)
	robotgo.Move(boundsX+boundsWidth/2, boundsY+boundsHeight/2)
}

// startTicker starts a ticker that updates the cursor at a given rate.
func (c *Client) startTicker(ctx context.Context) {
	ticker := time.NewTicker(time.Duration(1000/c.rate) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return

		case <-ticker.C:
			c.tick()
		}
	}
}

// tick updates the cursor after being called from the ticker.
func (c *Client) tick() {
	// If the cursor is not active, do nothing.
	if !c.active.Load() {
		return
	}

	// If the cursor is active, move the cursor in the active directions.
	c.step()
}

// step moves the cursor in the active directions.
func (c *Client) step() {
	// If the cursor is not active, do nothing.
	if !c.active.Load() {
		return
	}

	// Calculate the cursor movement.
	dx, dy := 0, 0
	if c.up.Load() {
		dy -= c.speed
	}
	if c.down.Load() {
		dy += c.speed
	}
	if c.left.Load() {
		dx -= c.speed
	}
	if c.right.Load() {
		dx += c.speed
	}

	// If the cursor is not moving, do nothing.
	if dx == 0 && dy == 0 {
		return
	}

	// Get the current cursor position.
	currX, currY := robotgo.Location()

	// Calculate the next cursor position.
	nextX, nextY := currX+dx, currY+dy

	// Get the display bounds.
	boundsX, boundsY, boundsWidth, boundsHeight := robotgo.GetDisplayBounds(robotgo.DisplayID)

	// Clamp the next cursor position to the display bounds.
	nextX = clamp(nextX, boundsX, boundsX+boundsWidth)
	nextY = clamp(nextY, boundsY, boundsY+boundsHeight)

	// Move the cursor to the new position.
	robotgo.Move(nextX, nextY)
}

// clamp clamps a value to a given range.
func clamp(value, min, max int) int {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
