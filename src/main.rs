use std::{cmp::max, iter::repeat, string};

struct NDArray {
    buf: Vec<i32>,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

fn compute_stride(shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .rev()
        .scan(1, |acc, &dim| {
            let prev = *acc;
            *acc *= dim;
            Some(prev)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    if a.len() != b.len() {
        return Err(format!("Shapes {:?} and {:?} are not broadcastable", a, b));
    }

    let out_dims = max(a.len(), b.len());
    let mut res = Vec::with_capacity(out_dims);

    for (aa, bb) in a.iter().zip(b.iter()).rev() {
        if aa == bb {
            res.push(*aa);
        } else if *aa == 1 {
            res.push(*bb)
        } else if *bb == 1 {
            res.push(*aa)
        } else {
            return Err(format!("Shapes {:?} and {:?} are not broadcastable", a, b));
        }
    }

    res.reverse();
    Ok(res)
}

struct MultiIndexIterator {
    shape: Vec<usize>,
    curr: Option<Vec<usize>>,
    done: bool,
}

impl MultiIndexIterator {
    fn new(shape: Vec<usize>) -> Self {
        let start = vec![0; shape.len()];
        MultiIndexIterator {
            shape: shape,
            curr: Some(start),
            done: false,
        }
    }
}

impl Iterator for MultiIndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let res = self.curr.as_ref().unwrap().clone();

        for i in (0..self.shape.len()).rev() {
            if self.curr.as_ref().unwrap()[i] + 1 < self.shape[i] {
                self.curr.as_mut().unwrap()[i] += 1;
                for j in i + 1..self.shape.len() {
                    self.curr.as_mut().unwrap()[j] = 0;
                }
                return Some(res);
            }
        }

        self.done = true;
        Some(res)
    }
}

fn broadcast_to(
    buf: Vec<i32>,
    shape: Vec<usize>,
    target_shape: Vec<usize>,
) -> Result<Vec<i32>, String> {
    let ldiff = target_shape.len().checked_sub(shape.len()).ok_or_else(|| {
        format!(
            "target_shape length ({}) is smaller than shape length ({})",
            target_shape.len(),
            shape.len()
        )
    })?;
    let padded_shape: Vec<usize> = repeat(1).take(ldiff).chain(shape.into_iter()).collect();

    let mut repeat_factors = Vec::new();
    for (p_dim, t_dim) in padded_shape.iter().zip(target_shape.iter()) {
        if *p_dim == *t_dim {
            repeat_factors.push(1);
        } else if *p_dim == 1 {
            repeat_factors.push(*t_dim);
        } else {
            return Err(format!(
                "Shapes are not compatible for broadcasting at dimension: {} vs {}",
                p_dim, t_dim
            ));
        }
    }

    fn rpt(buf: &[i32], buf_shape: &[usize], repeat_factors: &[usize]) -> Vec<i32> {
        if buf_shape.is_empty() {
            return buf.to_vec();
        }

        let mut res = Vec::new();
        let repeats = repeat_factors[0];
        let inner_buf_size = buf_shape[1..].iter().product::<usize>();
        let mut chunks = buf.chunks(inner_buf_size);

        if buf_shape[0] == 1 {
            // Only one chunk to repeat
            let chunk = chunks.next().unwrap();
            for _ in 0..repeats {
                res.extend(rpt(chunk, &buf_shape[1..], &repeat_factors[1..]));
            }
        } else {
            // Process each chunk
            for chunk in chunks {
                let sub_res = rpt(chunk, &buf_shape[1..], &repeat_factors[1..]);
                for _ in 0..repeats {
                    res.extend(sub_res.clone());
                }
            }
        }

        res
    }

    let res = rpt(&buf, &padded_shape, &repeat_factors);
    Ok(res)
}

fn bslice(buf: Vec<usize>, shape: Vec<usize>, bidxs: Vec<usize>) {
    // TODO: slice along batch indices
}

impl NDArray {
    fn new(buf: Vec<i32>, shape: Vec<usize>) -> Self {
        NDArray {
            buf: buf,
            shape: shape.clone(),
            stride: compute_stride(&shape),
        }
    }

    fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![1; size], shape)
    }

    fn matmul(&self, other: NDArray) -> Result<Self, &str> {
        let a_shape = &self.shape;
        let b_shape = &other.shape;
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err("Invalid dimensions!");
        }

        println!("a_shape: {:?}", a_shape);
        println!("b_shape: {:?}", b_shape);

        let a_bdims = &a_shape[..a_shape.len() - 2];
        let b_bdims = &b_shape[..b_shape.len() - 2];

        let a_nbdims = &a_shape[a_shape.len() - 2..];
        let b_nbdims = &b_shape[b_shape.len() - 2..];

        let bc_shape = broadcast(a_bdims, b_bdims).unwrap();
        let a_bc_shape: Vec<usize> = bc_shape
            .iter()
            .cloned()
            .chain(a_nbdims.iter().cloned())
            .collect();
        let b_bc_shape: Vec<usize> = bc_shape
            .iter()
            .cloned()
            .chain(b_nbdims.iter().cloned())
            .collect();
        println!("bc_shape: {:?}", bc_shape);
        println!("a_bc_shape: {:?}", a_bc_shape);
        println!("b_bc_shape: {:?}", b_bc_shape);

        let a_bc = broadcast_to(self.buf.clone(), a_shape.clone(), a_bc_shape);
        let b_bc = broadcast_to(other.buf.clone(), b_shape.clone(), b_bc_shape);

        let m = a_shape[a_shape.len() - 2];
        let n = a_shape[a_shape.len() - 1];
        let p = b_shape[b_shape.len() - 1];
        println!("m: {:?}, n: {:?}, p: {:?}", m, n, p);

        let mut out_shape = bc_shape.clone();
        out_shape.push(m);
        out_shape.push(p);
        println!("out_shape: {:?}", out_shape);

        let mut out = NDArray::new(vec![0; out_shape.iter().product()], out_shape.clone());

        let multi_iter = MultiIndexIterator::new(bc_shape);
        for idxs in multi_iter.into_iter() {
            println!("iter-shape: {:?}", idxs);

            // TODO: slice along batch dimensions, perform 2D matmul
        }

        Ok(out)
    }

    fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(self.buf.len(), shape.iter().product());
        Self::new(self.buf.clone(), shape)
    }
}

fn main() {
    let a = NDArray::ones(vec![1, 2, 1, 3]);
    let b = NDArray::ones(vec![4, 2, 3, 4]);
    // (3, 2, 1, 4)
    let c = a.matmul(b).unwrap();
    println!("c: {:?}", c.shape);
    println!("c: {:?}", c.buf);
}
