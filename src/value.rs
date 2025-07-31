use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({:?})", self.0.borrow().data)
    }
}

impl Value {
    pub fn new(a: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueData::new(a))))
    }

    fn clone_ref(&self) -> Value {
        Value(self.0.clone())
    }

    fn aug_grad(&self, grad: f64) {
        self.0.borrow_mut().grad += grad;
    }

    pub fn val(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_val(&self, val: f64) {
        self.0.borrow_mut().data = val;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn descend(&self) {
        self.0.borrow_mut().data = self.val() - 0.05 * self.grad();
    }

    fn postorder(&self, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
        let id = Rc::as_ptr(&self.0) as usize;
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        for kid in &self.0.borrow().kids {
            kid.postorder(visited, topo);
        }

        topo.push(self.clone());
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut topo = Vec::new();
        self.postorder(&mut visited, &mut topo);

        for v in &topo {
            v.0.borrow_mut().grad = 0.0;
        }
        self.0.borrow_mut().grad = 1.0;

        for v in topo.into_iter().rev() {
            if let Some(op) = &v.0.borrow().op {
                op.augment_kids(&v.0.borrow().kids, v.0.borrow().grad, v.val());
            }
        }
    }
}

#[derive(Debug)]
struct ValueData {
    data: f64,
    grad: f64,
    op: Option<Oper>,
    kids: Vec<Value>,
}

impl ValueData {
    fn new(a: f64) -> Self {
        ValueData {
            data: a,
            grad: 0.0,
            op: None,
            kids: vec![],
        }
    }
}

#[derive(Debug)]
enum Oper {
    Sum,
    Mul,
    Pow { exp: f64 },
    Relu,
    Tanh,
    Exp,
}

impl Oper {
    fn augment_kids(&self, children: &[Value], grad: f64, val: f64) {
        match self {
            Oper::Sum => {
                for child in children {
                    child.aug_grad(grad);
                }
            }
            Oper::Mul => {
                if children.len() == 2 {
                    children[0].aug_grad(children[1].val() * grad);
                    children[1].aug_grad(children[0].val() * grad);
                }
            }
            Oper::Pow { exp } => {
                if !children.is_empty() {
                    let base = &children[0];
                    base.aug_grad(exp * base.val().powf(exp - 1.0) * grad);
                }
            }
            Oper::Relu => {
                if !children.is_empty() {
                    let a = &children[0];
                    if a.val() > 0.0 {
                        a.aug_grad(grad);
                    }
                }
            }
            Oper::Tanh => {
                if !children.is_empty() {
                    let a = &children[0];
                    a.aug_grad(grad * (1.0 - (val * val)));
                }
            }
            Oper::Exp => {
                if !children.is_empty() {
                    let a = &children[0];
                    a.aug_grad(grad * val);
                }
            }
        }
    }
}

impl Display for Oper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Oper::Sum => write!(f, "+"),
            Oper::Mul => write!(f, "*"),
            Oper::Pow { exp } => write!(f, "^{}", exp),
            Oper::Relu => write!(f, "relu"),
            Oper::Tanh => write!(f, "tanh"),
            Oper::Exp => write!(f, "exp"),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.borrow().op.is_none() {
            write!(f, "{}", self.0.borrow().data)
        } else {
            write!(f, "{}", self.0.borrow().op.as_ref().unwrap())
        }
    }
}

impl Display for ValueData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Sum),
            kids: vec![self.clone_ref(), other.clone_ref()],
        })))
    }
}

impl AddAssign<&Value> for Value {
    fn add_assign(&mut self, other: &Self) {
        *self = &self.clone_ref() + other;
    }
}

impl Add<f64> for &Value {
    type Output = Value;

    fn add(self, other: f64) -> Self::Output {
        self + &Value::new(other)
    }
}

impl Add<&Value> for f64 {
    type Output = Value;

    fn add(self, other: &Value) -> Self::Output {
        &Value::new(self) + other
    }
}

impl Mul<&Value> for f64 {
    type Output = Value;

    fn mul(self, other: &Value) -> Self::Output {
        &Value::new(self) * other
    }
}

impl Mul<f64> for &Value {
    type Output = Value;

    fn mul(self, other: f64) -> Self::Output {
        self * &Value::new(other)
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data - other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Sum),
            kids: vec![self.clone_ref(), -&other.clone_ref()],
        })))
    }
}

impl Sub<f64> for &Value {
    type Output = Value;

    fn sub(self, other: f64) -> Self::Output {
        self - &Value::new(other)
    }
}

impl Sub<&Value> for f64 {
    type Output = Value;

    fn sub(self, other: &Value) -> Self::Output {
        &Value::new(self) - other
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Mul),
            kids: vec![self.clone_ref(), other.clone_ref()],
        })))
    }
}

impl Div<f64> for &Value {
    type Output = Value;

    fn div(self, other: f64) -> Self::Output {
        self * &Value::new(1.0 / other)
    }
}

impl Div<&Value> for f64 {
    type Output = Value;

    fn div(self, other: &Value) -> Self::Output {
        &Value::new(self) * &other.reciprocal()
    }
}

impl Div for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Self::Output {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data / other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Mul),
            kids: vec![self.clone_ref(), other.clone_ref().reciprocal()],
        })))
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        &Value::new(-1.0) * self
    }
}

impl Value {
    pub fn pow(&self, exp: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data.powf(exp),
            grad: 0.0,
            op: Some(Oper::Pow { exp }),
            kids: vec![self.clone_ref()],
        })))
    }

    pub fn reciprocal(&self) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data: 1.0 / self.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Pow { exp: -1.0 }),
            kids: vec![self.clone_ref()],
        })))
    }

    pub fn relu(&self) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data.max(0.0),
            grad: 0.0,
            op: Some(Oper::Relu),
            kids: vec![self.clone_ref()],
        })))
    }

    pub fn tanh(&self) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data.tanh(),
            grad: 0.0,
            op: Some(Oper::Tanh),
            kids: vec![self.clone_ref()],
        })))
    }

    pub fn exp(&self) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data.exp(),
            grad: 0.0,
            op: Some(Oper::Exp),
            kids: vec![self.clone_ref()],
        })))
    }
}
