# COBS-Rust
Rust implemntation of Constrained B-Splines


## What the Project Is About

**COBS-Rust** is a Rust library implementing Constrained B-Splines, a mathematical technique used for curve fitting and modeling with the ability to enforce constraints (such as smoothness, continuity, or boundary conditions). This is valuable in fields like data fitting, signal processing, computer graphics, and numerical optimization.

- The library is written in Rust.
- It uses the ndarray crate for handling multi-dimensional arrays and clarabel for solving optimization problems, which are typical in constrained curve fitting.
- The code is modular, with components for core algorithms, error handling, and possibly bindings for interoperability.

## How to Use

### 1. Add to Your Project

To include COBS-Rust in your own Rust project, add this to your `Cargo.toml` dependencies (once published, or via git if you use it directly):

```toml
[dependencies]
cobs_rs = { path = "path/to/cobs_rs" }
```

Or, if publishing to crates.io, use:

```toml
[dependencies]
cobs_rs = "0.1.0"
```

### 2. Basic Example

*(A real example would be based on the public API in core.rs. If you provide the core.rs file, I can create this sample. For now, here's a placeholder example.)*

```rust
use cobs_rs::core::*;

// Example: Constructing and evaluating a constrained B-spline
// let spline = ConstrainedBSpline::new(...);
// let value = spline.evaluate(t);
```

### 3. Running Tests

To run the tests for the library, navigate to the `cobs_rs` directory and run:

```sh
cargo test
```

### 4. Running Benchmarks

Benchmarking is available via Criterion. To run benchmarks:

```sh
cargo bench
```

## Features

- Construction and evaluation of constrained B-splines
- Constraint handling (equality and/or inequality constraints)
- Numerical stability and performance using ndarray and clarabel

## Development

- The core logic is in `cobs_rs/src/core.rs`.
- Error types are in `cobs_rs/src/error.rs`.
- (Bindings and additional interfaces may be available in `cobs_rs/src/bindings.rs`.)

## Requirements

- Rust 2021 edition
- See `Cargo.toml` for dependencies

---

### README.md Adaptation

Here is a revised README.md for your project:

```markdown
# COBS-Rust

A Rust implementation of Constrained B-Splines for curve fitting and modeling with constraints.

## Features

- Construction and evaluation of constrained B-Splines
- Support for constraint handling (equality/inequality)
- Utilizes `ndarray` for numerical arrays and `clarabel` for optimization

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
cobs_rs = { path = "path/to/cobs_rs" }
```

Example (pseudo-code):

```rust
use cobs_rs::core::*;

// Example: Constructing and evaluating a constrained B-spline
// let spline = ConstrainedBSpline::new(...);
// let value = spline.evaluate(t);
```

## Testing

Run tests:

```sh
cargo test
```

Run benchmarks:

```sh
cargo bench
```

## Structure

- `core.rs`: Main algorithms and logic for constrained B-splines
- `error.rs`: Error types and handling
- `bindings.rs`: (Optional) Interoperability with other systems/languages

## Requirements

- Rust 2021 edition
- See `Cargo.toml` for dependencies

---

*For more details and API usage, see the documentation or explore the source code.*
```

If you provide the contents of `core.rs`, I can adapt the usage example to be concrete and accurate. Let me know if you want to include more specific usage or API details!
