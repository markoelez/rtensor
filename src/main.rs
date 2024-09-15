use std::cmp::max;
use std::env;
use std::iter::repeat;
use std::process;

#[derive(Debug, Clone)]
struct NDArray {
    buf: Vec<i32>,
    shape: Vec<usize>,
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

    for (aa, bb) in a.iter().rev().zip(b.iter().rev()) {
        match (*aa, *bb) {
            (x, y) if x == y => res.push(x),
            (1, y) => res.push(y),
            (x, 1) => res.push(x),
            _ => return Err(format!("Shapes {:?} and {:?} are not broadcastable", a, b)),
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
        let curr = self.curr.as_mut()?;
        let res = curr.clone();

        for i in (0..self.shape.len()).rev() {
            if curr[i] + 1 < self.shape[i] {
                curr[i] += 1;
                curr[i + 1..].fill(0);
                return Some(res);
            }
        }

        self.curr = None;
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

    let padded_shape: Vec<usize> = repeat(1).take(ldiff).chain(shape).collect();

    let repeat_factors: Vec<usize> = padded_shape
        .iter()
        .zip(&target_shape)
        .map(|(&p_dim, &t_dim)| {
            if p_dim == t_dim {
                Ok(1)
            } else if p_dim == 1 {
                Ok(t_dim)
            } else {
                Err(format!(
                    "Shapes are not compatible for broadcasting at dimension: {} vs {}",
                    p_dim, t_dim
                ))
            }
        })
        .collect::<Result<_, _>>()?;

    Ok(rpt(&buf, &padded_shape, &repeat_factors))
}

fn rpt(buf: &[i32], buf_shape: &[usize], repeat_factors: &[usize]) -> Vec<i32> {
    if buf_shape.is_empty() {
        return buf.to_vec();
    }

    let repeats = repeat_factors[0];
    let inner_buf_size: usize = buf_shape[1..].iter().product();
    let mut chunks = buf.chunks(inner_buf_size);

    if buf_shape[0] == 1 {
        // Only one chunk to repeat
        chunks
            .next()
            .map(|chunk| {
                (0..repeats)
                    .flat_map(|_| rpt(chunk, &buf_shape[1..], &repeat_factors[1..]))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        // Process each chunk
        chunks
            .flat_map(|chunk| {
                let sub_res = rpt(chunk, &buf_shape[1..], &repeat_factors[1..]);
                repeat(sub_res).take(repeats).flatten()
            })
            .collect()
    }
}

fn bslice<'a>(buf: &'a [i32], shape: &[usize], bidxs: &[usize]) -> &'a [i32] {
    let strides = compute_stride(shape);
    let offset = bidxs
        .iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx * stride)
        .sum::<usize>();

    let slice_size = shape[bidxs.len()..].iter().product::<usize>();
    &buf[offset..offset + slice_size]
}

impl NDArray {
    fn new(buf: Vec<i32>, shape: Vec<usize>) -> Self {
        NDArray {
            buf: buf,
            shape: shape.clone(),
        }
    }

    fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![1; size], shape)
    }

    fn matmul(&self, other: &NDArray) -> Result<Self, String> {
        let (a_shape, b_shape) = (&self.shape, &other.shape);
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err("Invalid dimensions!".to_string());
        }

        let (a_bdims, b_bdims) = (&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2]);
        let (a_nbdims, b_nbdims) = (&a_shape[a_shape.len() - 2..], &b_shape[b_shape.len() - 2..]);

        let bc_shape = broadcast(a_bdims, b_bdims)?;
        let a_bc_shape: Vec<usize> = bc_shape.iter().chain(a_nbdims.iter()).cloned().collect();
        let b_bc_shape: Vec<usize> = bc_shape.iter().chain(b_nbdims.iter()).cloned().collect();

        let a_bc = broadcast_to(self.buf.clone(), a_shape.clone(), a_bc_shape.clone())?;
        let b_bc = broadcast_to(other.buf.clone(), b_shape.clone(), b_bc_shape.clone())?;

        let (m, n, p) = (a_nbdims[0], a_nbdims[1], b_nbdims[1]);

        let mut out_shape = bc_shape.clone();
        out_shape.extend_from_slice(&[m, p]);

        let mut out = NDArray::new(vec![0; out_shape.iter().product()], out_shape.clone());

        let multi_iter = MultiIndexIterator::new(bc_shape);
        for idxs in multi_iter {
            let a_mat = bslice(&a_bc, &a_bc_shape, &idxs);
            let b_mat = bslice(&b_bc, &b_bc_shape, &idxs);

            let mut c_mat = vec![0; m * p];
            for i in 0..m {
                for j in 0..p {
                    c_mat[i * p + j] = (0..n).map(|k| a_mat[i * n + k] * b_mat[k * p + j]).sum();
                }
            }

            let out_strides = compute_stride(&out.shape);
            let out_offset = idxs
                .iter()
                .zip(out_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum::<usize>();
            let c_mat_len = m * p;
            out.buf[out_offset..out_offset + c_mat_len].copy_from_slice(&c_mat);
        }

        Ok(out)
    }
}

fn parse_array(s: &str) -> Result<NDArray, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err("Invalid array format".to_string());
    }

    let shape: Vec<usize> = parts[0]
        .split(',')
        .map(|s| s.parse().map_err(|_| "Invalid shape".to_string()))
        .collect::<Result<_, _>>()?;

    let buf: Vec<i32> = parts[1]
        .split(',')
        .map(|s| s.parse().map_err(|_| "Invalid buffer".to_string()))
        .collect::<Result<_, _>>()?;

    if buf.len() != shape.iter().product() {
        return Err("Buffer size doesn't match shape".to_string());
    }

    Ok(NDArray::new(buf, shape))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <array1> <array2>", args[0]);
        eprintln!("Format: shape1,shape2,...:val1,val2,...");
        process::exit(1);
    }

    let a = parse_array(&args[1]).unwrap_or_else(|e| {
        eprintln!("Error parsing first array: {}", e);
        process::exit(1);
    });

    let b = parse_array(&args[2]).unwrap_or_else(|e| {
        eprintln!("Error parsing second array: {}", e);
        process::exit(1);
    });

    match a.matmul(&b) {
        Ok(result) => {
            println!(
                "{}:{}",
                result
                    .shape
                    .iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                result
                    .buf
                    .iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        Err(e) => {
            eprintln!("Error during matrix multiplication: {}", e);
            process::exit(1);
        }
    }
}
