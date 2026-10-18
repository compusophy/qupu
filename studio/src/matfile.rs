//! Minimal MATLAB level-5 MAT-file reader: enough for foundry measurement
//! files (structs of numeric arrays, optionally zlib-compressed).
//!
//! Format reference: MathWorks "MAT-File Format", level 5. Each data element
//! is an 8-byte tag (type, byte count) followed by its body padded to 8
//! bytes, or a "small element" packed into the tag itself.

use miniz_oxide::inflate::decompress_to_vec_zlib;

#[derive(Debug, Clone)]
pub enum Value {
    Struct(Vec<(String, Value)>),
    Numeric { dims: Vec<usize>, data: Vec<f64> },
    Other,
}

impl Value {
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Struct(fields) => fields.iter().find(|(n, _)| n == name).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn path(&self, path: &[&str]) -> Option<&Value> {
        path.iter().try_fold(self, |v, p| v.field(p))
    }

    pub fn numbers(&self) -> Option<&[f64]> {
        match self {
            Value::Numeric { data, .. } => Some(data),
            _ => None,
        }
    }
}

const MI_INT8: u32 = 1;
const MI_INT32: u32 = 5;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MX_STRUCT: u32 = 2;

/// Parse a MAT file into its top-level named variables.
pub fn parse(bytes: &[u8]) -> Result<Vec<(String, Value)>, String> {
    if bytes.len() < 128 || &bytes[126..128] != b"IM" {
        return Err("not a little-endian level-5 MAT file".into());
    }
    let mut out = Vec::new();
    let mut off = 128;
    while off + 8 <= bytes.len() {
        let (ty, n) = (u32_at(bytes, off), u32_at(bytes, off + 4) as usize);
        let body = bytes.get(off + 8..off + 8 + n).ok_or("truncated element")?;
        match ty {
            MI_COMPRESSED => {
                let inner = decompress_to_vec_zlib(body).map_err(|e| format!("zlib: {e:?}"))?;
                let (ity, inn) = (u32_at(&inner, 0), u32_at(&inner, 4) as usize);
                if ity == MI_MATRIX {
                    out.push(matrix(&inner[8..8 + inn])?);
                }
            }
            MI_MATRIX => out.push(matrix(body)?),
            _ => {}
        }
        off += 8 + pad8(n);
    }
    Ok(out)
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// Read one sub-element at `o`; returns (type, payload, offset of the next).
fn element(b: &[u8], o: usize) -> Result<(u32, &[u8], usize), String> {
    if o + 8 > b.len() {
        return Err("truncated sub-element".into());
    }
    let tag = u32_at(b, o);
    if tag >> 16 != 0 {
        // Small data element: type in the low 16 bits, size in the high 16.
        let n = (tag >> 16) as usize;
        Ok((tag & 0xffff, &b[o + 4..o + 4 + n], o + 8))
    } else {
        let n = u32_at(b, o + 4) as usize;
        let p = b.get(o + 8..o + 8 + n).ok_or("truncated payload")?;
        Ok((tag, p, o + 8 + pad8(n)))
    }
}

/// Body of an miMATRIX element → (name, value).
fn matrix(b: &[u8]) -> Result<(String, Value), String> {
    let (_, flags, o) = element(b, 0)?;
    let class = u32_at(flags, 0) & 0xff;
    let (_, dims_raw, o) = element(b, o)?;
    let dims: Vec<usize> = dims_raw
        .chunks_exact(4)
        .map(|c| u32_at(c, 0) as usize)
        .collect();
    let (_, name_raw, mut o) = element(b, o)?;
    let name = String::from_utf8_lossy(name_raw)
        .trim_end_matches('\0')
        .to_string();

    if class == MX_STRUCT {
        let (_, len_raw, o2) = element(b, o)?;
        let len = u32_at(len_raw, 0) as usize;
        let (ty, names_raw, o3) = element(b, o2)?;
        if ty != MI_INT8 || len == 0 {
            return Err("bad struct field names".into());
        }
        let names: Vec<String> = names_raw
            .chunks(len)
            .map(|c| {
                String::from_utf8_lossy(c)
                    .trim_end_matches('\0')
                    .to_string()
            })
            .collect();
        o = o3;
        let mut fields = Vec::with_capacity(names.len());
        for n in names {
            let (ty, body, next) = element(b, o)?;
            if ty != MI_MATRIX {
                return Err("struct field is not a matrix".into());
            }
            let (_, v) = if body.is_empty() {
                (String::new(), Value::Other)
            } else {
                matrix(body)?
            };
            fields.push((n, v));
            o = next;
        }
        return Ok((name, Value::Struct(fields)));
    }

    // Numeric classes 6..=15 (double, single, int8..uint64): real part only.
    if (6..=15).contains(&class) {
        let (ty, raw, _) = element(b, o)?;
        return Ok((
            name,
            Value::Numeric {
                dims,
                data: numbers(ty, raw)?,
            },
        ));
    }
    Ok((name, Value::Other))
}

/// MATLAB may store a double array in a narrower integer type; widen it.
fn numbers(ty: u32, raw: &[u8]) -> Result<Vec<f64>, String> {
    let v = match ty {
        1 => raw.iter().map(|&x| x as i8 as f64).collect(),
        2 => raw.iter().map(|&x| x as f64).collect(),
        3 => raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        4 => raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f64)
            .collect(),
        MI_INT32 => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        6 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        7 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        9 => raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        12 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        13 => raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        _ => return Err(format!("unsupported numeric storage type {ty}")),
    };
    Ok(v)
}
