# karp

A compact Rust port inspired by Andrej Karpathy’s “micrograd/autograd” tutorials. This crate implements a tiny reverse‑mode automatic differentiation engine (autograd) with a minimal tensor/value type, a computation graph, and gradient backpropagation.

## Background: following Karpathy’s tutorial

This project follows the core ideas in Andrej Karpathy’s micrograd series:

- Build a tiny scalar/tensor `Value` type that tracks data and gradient
- Record a directed acyclic computation graph of primitive ops
- Perform reverse‑mode autodiff by traversing the graph in topological order

While Karpathy’s original reference code is in Python, this implementation uses idiomatic Rust while keeping the design intentionally small and readable.

References:

- micrograd (Python): https://github.com/karpathy/micrograd
- Lecture/Tutorial video(s): https://www.youtube.com/watch?v=VMj-3S1tku0

## What is autograd? (Reverse‑mode autodiff, briefly)

Given a scalar loss L computed from inputs via a composition of elementary ops, reverse‑mode autodiff computes dL/dx for every intermediate x efficiently:

1. Forward pass: evaluate each node once, storing the result (and any saved context needed for gradients).
2. Backward pass: start from dL/dL = 1 at the loss, then propagate gradients to parents using the local derivatives from the chain rule.
3. Order: visit nodes in reverse topological order so children’s gradients are known before parents are processed.

Why reverse mode? When you have one (or few) scalar outputs and many inputs/parameters, reverse mode gives you all dL/d(theta) in about the cost of a few forward passes, which is perfect for ML training.

### Algorithm sketch

- Each node holds: value (data), grad (initialized to 0), parents, and a backward function that distributes its gradient to parents.
- To backprop:
    - Set loss.grad = 1
    - Build a topological order of nodes reachable from loss to ensure parent gradients are comprehensive
    - Traverse in reverse order; call each node’s backward to accumulate parent grads

## The reference/ownership model used here (and why)

Rust’s ownership rules make dynamic computation graphs trickier than in Python. This implementation uses reference‑counted pointers to express shared graph topology safely at runtime:

- Rc/Arc for shared ownership of nodes (graph edges may point to the same parent multiple times). In single‑threaded builds, `Rc<RefCell<Node>>` is typical; for thread‑safe variants, `Arc<Mutex<Node>>` could be used.
- Interior mutability (RefCell) to allow:
    - Accumulating `grad` during backprop even when multiple children update the same parent
    - Storing closures/backward fns that mutate parent gradients
- Weak references (Weak) to avoid cycles where appropriate (e.g., parent lists) so nodes can be dropped when no longer needed.

## Project layout

- `src/value.rs` — the core `Value`/node type, ops, and backprop logic
- `src/main.rs` — small multi-layer perceptron demo, using the backprop logic type to perform gradient descent and learn to produce contrived outputs for hard-coded inputs.

## Usage

Build and run the pseudo-demo:

```bash
cargo run --debug
```

Should output:

```
[src/main.rs:41:5] &trial_ys = [
    Value(0.99...),
    Value(-0.99...),
    Value(-0.99...),
    Value(0.99...),
]
```

This output is indicative of the fact that the model has learned to predict 1, -1, -1, 1 for its 4 input vectors.

## Add as a library in another crate (example `Cargo.toml`):

```toml
[dependencies]
karp = { path = "../karp" }
```

Then, in your code:

```rust
use karp::Value;

fn main() {
    let x = Value::from(2.0);
    let y = Value::from(-3.0);
    let z = &x * &y + &x; // example ops
    z.backward();
    println!("z={} dz/dx={} dz/dy={}", z.data(), x.grad(), y.grad());
}
```


## Extending

- Add more primitives (sigmoid, etc)
- Vector/matrix containers or a tiny Tensor wrapper
- Optimizers (SGD, Adam) and simple training loops
- Serialization of graphs or parameters
