# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./

# Allow Go to auto-download the correct toolchain version if needed
ENV GOTOOLCHAIN=auto
RUN go mod download

# Copy source
COPY . .

# Build binary (GOTOOLCHAIN=auto inherited from ENV above)
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o server ./cmd/server

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates wget

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/server .

# Create data directory
RUN mkdir -p /data/wal

EXPOSE 8080

# Run as non-root
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser && \
    chown -R appuser:appuser /app /data

USER appuser

CMD ["./server"]
