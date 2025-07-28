#![allow(unused)]
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Debug, Clone)]
struct ValueRef(Rc<RefCell<Value>>);

impl ValueRef {
    fn new(a: f32) -> Self {
        ValueRef(Rc::new(RefCell::new(Value::new(a))))
    }

    fn clone_ref(&self) -> ValueRef {
        ValueRef(self.0.clone())
    }

    fn aug_grad(&self, grad: f32) {
        self.0.borrow_mut().grad += grad;
    }

    fn val(&self) -> f32 {
        self.0.borrow().data
    }

    fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    fn get_kids(&self) -> Vec<ValueRef> {
        if let Some(oper) = self.0.borrow().op.as_ref() {
            return oper.kids();
        } else {
            return vec![];
        }
    }

    fn postorder(&self, visited: &mut HashSet<usize>, topo: &mut Vec<ValueRef>) {
        let id = Rc::as_ptr(&self.0) as usize;
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        if let Some(op) = &self.0.borrow().op {
            for kid in op.kids() {
                kid.postorder(visited, topo);
            }
        }

        topo.push(self.clone());
    }

    fn backward(&self) {
        let mut visited = HashSet::new();
        let mut topo = Vec::new();
        self.postorder(&mut visited, &mut topo);

        for v in &topo {
            v.0.borrow_mut().grad = 0.0;
        }
        self.0.borrow_mut().grad = 1.0;

        for v in topo.into_iter().rev() {
            if let Some(op) = &v.0.borrow().op {
                op.augment_kids(v.0.borrow().grad);
            }
        }
    }
}

fn main() {
    // println!("Hello, world!");
    let a: ValueRef = ValueRef::new(-4.0);
    let b: ValueRef = ValueRef::new(2.0);
    let mut c = a.clone() + b.clone();
    let mut d = a.clone() * b.clone() + b.clone().pow(3.0);
    c += c.clone() + 1.0;
    c += 1.0 + c.clone() + (-a.clone());
    d += d.clone() * 2.0 + (b.clone() + a.clone()).relu();
    d += 3.0 * d.clone() + (b.clone() - a.clone()).relu();
    let e = c - d;
    let f = e.clone().pow(2.0);
    let mut g = f.clone() / 2.0;
    g += 10.0 / f;

    g.0.borrow_mut().grad = 1.0;
    g.backward();

    // println!("{}", g);
    println!("{}", g.0.borrow().data);

    println!("{}", a.grad());
    println!("{}", b.grad());
}

#[derive(Debug)]
struct Value {
    data: f32,
    grad: f32,
    op: Option<Oper>,
}

impl Value {
    fn new(a: f32) -> Self {
        Value {
            data: a,
            grad: 0.0,
            op: None,
        }
    }
}

#[derive(Debug)]
enum Oper {
    Sum { a: ValueRef, b: ValueRef },
    SumF { a: ValueRef, b: f32 },
    Mul { a: ValueRef, b: ValueRef },
    MulF { a: ValueRef, b: f32 },
    Pow { base: ValueRef, exp: f32 },
    Relu { a: ValueRef },
}

impl Oper {
    fn augment_kids(&self, grad: f32) {
        match self {
            Oper::Sum { a, b } => {
                a.aug_grad(grad);
                b.aug_grad(grad);
            }
            Oper::SumF { a, b: _ } => {
                a.aug_grad(grad);
            }
            Oper::Mul { a, b } => {
                a.aug_grad(b.val() * grad);
                b.aug_grad(a.val() * grad);
            }
            Oper::MulF { a, b } => {
                a.aug_grad(b * grad);
            }
            Oper::Pow { base, exp } => {
                base.aug_grad(exp * base.val().powf(exp - 1.0) * grad);
            }
            Oper::Relu { a } => {
                if a.val() > 0.0 {
                    a.aug_grad(grad);
                }
            }
        }
    }

    fn kids(&self) -> Vec<ValueRef> {
        match self {
            Oper::Sum { a, b } => {
                vec![a.clone(), b.clone()]
            }
            Oper::SumF { a, b: _ } => {
                vec![a.clone()]
            }
            Oper::Mul { a, b } => {
                vec![a.clone(), b.clone()]
            }
            Oper::MulF { a, b } => {
                vec![a.clone()]
            }
            Oper::Pow { base, exp } => {
                vec![base.clone()]
            }
            Oper::Relu { a } => {
                vec![a.clone()]
            }
        }
    }
}

impl Display for Oper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Oper::Sum { a, b } => write!(f, "+ {} {}", a, b),
            Oper::SumF { a, b } => write!(f, "+ {} {}", a, b),
            Oper::Mul { a, b } => write!(f, "* {} {}", a, b),
            Oper::MulF { a, b } => write!(f, "* {} {}", a, b),
            Oper::Pow { base, exp } => write!(f, "^ {} {}", base, exp),
            // Oper::Neg { a } => write!(f, "-{}", a),
            // Oper::Sub { a, b } => write!(f, "({} - {})", a, b),
            // Oper::SubF { a, b } => write!(f, "({} - {})", a, b),
            // Oper::DivTop { a, b } => write!(f, "({} / {})", a, b),
            // Oper::DivBot { a, b } => write!(f, "({} / {})", a, b),
            Oper::Relu { a } => write!(f, "relu {}", a),
        }
    }
}

impl Display for ValueRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.borrow().op.is_none() {
            write!(f, "{}", self.0.borrow().data)
        } else {
            write!(f, "{}", self.0.borrow().op.as_ref().unwrap())
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl Add for ValueRef {
    type Output = ValueRef;

    fn add(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Sum {
                a: ValueRef(self.0.clone()),
                b: ValueRef(other.0.clone()),
            }),
        })))
    }
}

impl AddAssign for ValueRef {
    fn add_assign(&mut self, other: ValueRef) {
        *self = self.clone_ref() + other;
    }
}

impl Add<f32> for ValueRef {
    type Output = ValueRef;

    fn add(self, other: f32) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data + other,
            grad: 0.0,
            op: Some(Oper::SumF {
                a: self.clone_ref(),
                b: other,
            }),
        })))
    }
}

impl Add<ValueRef> for f32 {
    type Output = ValueRef;

    fn add(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self + other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::SumF {
                a: other.clone_ref(),
                b: self,
            }),
        })))
    }
}

impl Mul<ValueRef> for f32 {
    type Output = ValueRef;

    fn mul(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self * other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::MulF {
                a: other.clone_ref(),
                b: self,
            }),
        })))
    }
}

impl Mul<f32> for ValueRef {
    type Output = ValueRef;

    fn mul(self, other: f32) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data * other,
            grad: 0.0,
            op: Some(Oper::MulF {
                a: self.clone_ref(),
                b: other,
            }),
        })))
    }
}

impl Sub for ValueRef {
    type Output = ValueRef;

    fn sub(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data - other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Sum {
                a: self.clone_ref(),
                b: -other.clone_ref(),
            }),
        })))
    }
}

impl Sub<f32> for ValueRef {
    type Output = ValueRef;

    fn sub(self, other: f32) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data - other,
            grad: 0.0,
            op: Some(Oper::SumF {
                a: self.clone_ref(),
                b: -other,
            }),
        })))
    }
}

impl Sub<ValueRef> for f32 {
    type Output = ValueRef;

    fn sub(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self - other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::SumF {
                a: -other.clone_ref(),
                b: self,
            }),
        })))
    }
}

impl Mul for ValueRef {
    type Output = ValueRef;

    fn mul(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Mul {
                a: self.clone_ref(),
                b: other.clone_ref(),
            }),
        })))
    }
}

// impl Mul<f32> for ValueRef {
//     type Output = ValueRef;

//     fn mul(self, other: f32) -> Self::Output {
//         ValueRef(Rc::new(RefCell::new(Value {
//             data: self.0.borrow().data * other,
//             grad: 0.0,
//             op: Some(Oper::MulF {
//                 a: self.borrow(),
//                 b: other,
//             }),
//         })))
//     }
// }

impl Div<f32> for ValueRef {
    type Output = ValueRef;

    fn div(self, other: f32) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data / other,
            grad: 0.0,
            op: Some(Oper::MulF {
                a: self.clone_ref(),
                b: 1.0 / other,
            }),
        })))
    }
}

impl Div<ValueRef> for f32 {
    type Output = ValueRef;

    fn div(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self / other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::MulF {
                a: other.clone_ref().reciprocal(),
                b: self,
            }),
        })))
    }
}

impl Div<ValueRef> for ValueRef {
    type Output = ValueRef;

    fn div(self, other: ValueRef) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data / other.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Mul {
                a: self.clone_ref(),
                b: other.clone_ref().reciprocal(),
            }),
        })))
    }
}

impl Neg for ValueRef {
    type Output = ValueRef;

    fn neg(self) -> Self::Output {
        ValueRef(Rc::new(RefCell::new(Value {
            data: -self.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::MulF {
                a: self.clone_ref(),
                b: -1.0,
            }),
        })))
    }
}

impl ValueRef {
    pub fn pow(self, exp: f32) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data.powf(exp),
            grad: 0.0,
            op: Some(Oper::Pow {
                base: self.clone_ref(),
                exp,
            }),
        })))
    }

    pub fn reciprocal(self) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data: 1.0 / self.0.borrow().data,
            grad: 0.0,
            op: Some(Oper::Pow {
                base: self.clone_ref(),
                exp: -1.0,
            }),
        })))
    }

    pub fn relu(self) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data: self.0.borrow().data.max(0.0),
            grad: 0.0,
            op: Some(Oper::Relu {
                a: self.clone_ref(),
            }),
        })))
    }
}
