package wal

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// InboxWAL provides write-ahead logging for incoming PCS requests
type InboxWAL struct {
	mu   sync.Mutex
	file *os.File
	path string
}

// Entry represents a single WAL entry
type Entry struct {
	Timestamp time.Time
	Body      []byte
}

// NewInboxWAL creates or opens an inbox WAL file
func NewInboxWAL(dirPath string) (*InboxWAL, error) {
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	walPath := filepath.Join(dirPath, fmt.Sprintf("inbox-%s.wal", time.Now().Format("20060102")))

	file, err := os.OpenFile(walPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL file: %w", err)
	}

	return &InboxWAL{
		file: file,
		path: walPath,
	}, nil
}

// Append writes a request body to the WAL with fsync
func (w *InboxWAL) Append(body []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	entry := Entry{
		Timestamp: time.Now(),
		Body:      body,
	}

	// Write timestamp + body length + body
	line := fmt.Sprintf("%s|%d|%s\n", entry.Timestamp.Format(time.RFC3339Nano), len(body), body)

	if _, err := w.file.WriteString(line); err != nil {
		return fmt.Errorf("failed to write WAL entry: %w", err)
	}

	// Critical: fsync to ensure durability
	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("failed to sync WAL: %w", err)
	}

	return nil
}

// Close flushes and closes the WAL
func (w *InboxWAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.file.Sync(); err != nil {
		return err
	}
	return w.file.Close()
}

// Replay reads all entries from a WAL file
func Replay(walPath string) ([]Entry, error) {
	file, err := os.Open(walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer file.Close()

	var entries []Entry
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		// Parse: timestamp|length|body
		var ts string
		var length int
		var body string

		n, err := fmt.Sscanf(line, "%[^|]|%d|%s", &ts, &length, &body)
		if err != nil || n != 3 {
			continue // skip malformed lines
		}

		timestamp, err := time.Parse(time.RFC3339Nano, ts)
		if err != nil {
			continue
		}

		entries = append(entries, Entry{
			Timestamp: timestamp,
			Body:      []byte(body),
		})
	}

	return entries, scanner.Err()
}

// RotateWAL creates a new daily WAL file and returns old file path
func RotateWAL(dirPath string, currentWAL *InboxWAL) (*InboxWAL, string, error) {
	currentWAL.mu.Lock()
	oldPath := currentWAL.path
	currentWAL.mu.Unlock()

	if err := currentWAL.Close(); err != nil {
		return nil, "", fmt.Errorf("failed to close current WAL: %w", err)
	}

	newWAL, err := NewInboxWAL(dirPath)
	if err != nil {
		return nil, "", err
	}

	return newWAL, oldPath, nil
}
