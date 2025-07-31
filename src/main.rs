mod value;
use rand::Rng;
use std::fmt::Display;
use value::Value;
fn main() {
    let mlp = MLP::new(3, &[4, 4, 1]);
    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, 1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];

    let ys = [1.0, -1.0, -1.0, 1.0];

    for _ in 0..100000 {
        let trial_ys = xs
            .iter()
            .map(|x| mlp.call(&to_vals(x)).first().unwrap().clone())
            .collect::<Vec<Value>>();

        let loss = trial_ys
            .iter()
            .zip(&ys)
            .map(|(y, y_hat)| (y - &Value::new(*y_hat)).pow(2.0))
            .fold(Value::new(0.0), |acc, x| (&acc + &x));
        loss.backward();
        mlp.descend();
    }
    let trial_ys = xs
        .iter()
        .map(|x| mlp.call(&to_vals(x)).first().unwrap().clone())
        .collect::<Vec<Value>>();

    let loss = trial_ys
        .iter()
        .zip(&ys)
        .map(|(y, y_hat)| (y - &Value::new(*y_hat)).pow(2.0))
        .fold(Value::new(0.0), |acc, x| (&acc + &x));

    dbg!(&trial_ys);
    // println!("{}", mlp);
}

fn vals(vals: Vec<Value>) -> Vec<f64> {
    vals.iter().map(|v| v.val()).collect::<Vec<f64>>()
}

fn to_vals(floats: &[f64]) -> Vec<Value> {
    floats.iter().map(|&f| Value::new(f)).collect()
}

struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    fn new(nin: usize) -> Self {
        let weights = (0..nin)
            .map(|_| Value::new(rand::random_range(-1.0..1.0)))
            .collect();
        let bias = Value::new(rand::random_range(-1.0..1.0));
        Self { weights, bias }
    }

    fn call(&self, x: &[Value]) -> Value {
        assert_eq!(self.weights.len(), x.len());
        let mut out = self.bias.clone();
        for (i, weight) in self.weights.iter().enumerate() {
            out += &(weight * &x[i]);
        }
        out.tanh()
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        params.extend(&self.weights);
        params.push(&self.bias);
        params
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(nin: usize, nout: usize) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Self { neurons }
    }

    fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        for neuron in &self.neurons {
            params.extend(neuron.parameters());
        }
        params
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MLP {{ layers: [")?;
        for (i, layer) in self.layers.iter().enumerate() {
            write!(f, "{}", layer)?;
            if i < self.layers.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "] }}")
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer {{ neurons: [")?;
        for (i, neuron) in self.neurons.iter().enumerate() {
            write!(f, "{}", neuron)?;
            if i < self.neurons.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "] }}")
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neuron {{ weights: [")?;
        for (i, weight) in self.weights.iter().enumerate() {
            write!(f, "{}", weight)?;
            if i < self.weights.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "] }}")
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut layers = Vec::new();
        layers.push(Layer::new(nin, nouts[0]));
        for i in 1..nouts.len() {
            layers.push(Layer::new(nouts[i - 1], nouts[i]));
        }
        Self { layers }
    }

    fn call(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in &self.layers {
            out = layer.call(&out);
        }
        out
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn descend(&self) {
        for l in &self.layers {
            for n in &l.neurons {
                for w in &n.weights {
                    w.descend();
                }
            }
        }
    }
}
