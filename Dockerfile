# Base images are version-pinned rather than digest-pinned so the recipe stays
# readable; pin digests if you need bit-identical rebuilds years from now.
FROM rust:1.96.0-bookworm AS build
WORKDIR /artifact
# Cargo.lock is a release gate: this COPY fails the build if it is absent.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
RUN cargo build --locked --release

# 3.12, not 3.11: requirements.txt is compiled with --python-version 3.12 and
# carries --hash pins, so pip runs in hash-checking mode and cannot resolve a
# 3.11 wheel set. The recorded campaign toolchain is 3.12.3.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates build-essential cmake git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Every experiment tier shells out to `cargo run --locked --release`, and
# `reproduce.py doctor` checks for cargo and rustc, so the runtime stage needs a
# real toolchain -- not just the prebuilt binary. Carry it over from the build
# stage rather than re-running rustup.
COPY --from=build /usr/local/rustup /usr/local/rustup
COPY --from=build /usr/local/cargo /usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

WORKDIR /artifact
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=build /artifact/target/release/ultravec /usr/local/bin/ultravec
COPY . .
ENV ULTRAVEC_BIN=/usr/local/bin/ultravec PYTHONUNBUFFERED=1
CMD ["python3", "scripts/reproduce.py", "doctor"]
