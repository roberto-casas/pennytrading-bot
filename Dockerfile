FROM rust:1.85-slim AS builder

WORKDIR /app

# Install build dependencies for rusqlite bundled
RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*

# Cache dependency builds by copying manifests first
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN cargo build --release && rm -rf src target/release/pennytrading-bot

# Build the actual binary
COPY src/ src/
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/pennytrading-bot /usr/local/bin/pennytrading-bot
COPY config.yaml /etc/pennytrading-bot/config.yaml

ENTRYPOINT ["pennytrading-bot"]
