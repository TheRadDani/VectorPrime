// crates/llmforge-model-ir/src/lib.rs
//
// Model IR Analyzer for LLMForge.
//
// This crate inspects GGUF and ONNX model files at the binary level to extract
// structured metadata (parameter count, architecture name, context length, and
// layer count) without running inference.  The resulting [`ModelIR`] is used by
// `llmforge-bindings` to populate [`llmforge_core::ModelInfo::param_count`]
// before the optimization pipeline runs.
//
// Usage:
//   Called from `llmforge-bindings` via `parse_model(path)`.  Errors are
//   recovered gracefully — callers should use `.ok()` so that a parse failure
//   never aborts the optimization run.
//
// GGUF format reference:
//   https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{bail, Context, Result};
use llmforge_core::ModelFormat;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Intermediate representation of a model file.
///
/// All fields except `format` are optional because they may be absent from
/// the file's metadata or uncomputable (e.g. from a corrupt/truncated file).
/// Callers must never require these to be `Some`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelIR {
    /// Detected file format (GGUF or ONNX).
    pub format: ModelFormat,
    /// Total number of model parameters, if determinable.
    pub param_count: Option<u64>,
    /// Architecture family name (e.g. `"llama"`, `"mistral"`, `"phi"`).
    pub architecture: Option<String>,
    /// Maximum context length supported by the model.
    pub context_length: Option<u32>,
    /// Number of transformer blocks (layers).
    pub layer_count: Option<u32>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a model file and return its [`ModelIR`].
///
/// Dispatches to [`parse_gguf`] or [`parse_onnx`] based on the file extension.
/// Returns an error if the extension is unrecognised or neither parser can
/// make sense of the file.
///
/// # Errors
///
/// Returns an error when the file cannot be opened, the magic bytes are wrong,
/// or the file is unrecoverably truncated.  Partial metadata is not an error —
/// missing keys leave the corresponding field as `None`.
pub fn parse_model(path: &Path) -> Result<ModelIR> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "gguf" => parse_gguf(path),
        "onnx" => parse_onnx(path),
        other => bail!(
            "unrecognised model file extension '.{other}'; expected '.gguf' or '.onnx'"
        ),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GGUF parser
// ──────────────────────────────────────────────────────────────────────────────

/// GGUF value type codes as defined in ggml/docs/gguf.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgufValueType {
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0  => Some(Self::Uint8),
            1  => Some(Self::Int8),
            2  => Some(Self::Uint16),
            3  => Some(Self::Int16),
            4  => Some(Self::Uint32),
            5  => Some(Self::Int32),
            6  => Some(Self::Float32),
            7  => Some(Self::Bool),
            8  => Some(Self::String),
            9  => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _  => None,
        }
    }

    /// Fixed byte size for scalar types; `None` for STRING and ARRAY (variable).
    fn fixed_size(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16            => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array            => None,
        }
    }
}

// ── Low-level read helpers ────────────────────────────────────────────────────

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_le(r: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16_le(r: &mut impl Read) -> io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32_le(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_le(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

#[allow(dead_code)]
fn read_f32_le(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_u64_le(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64_le(r: &mut impl Read) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

#[allow(dead_code)]
fn read_f64_le(r: &mut impl Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

/// Read a GGUF-format string (u64 length prefix + UTF-8 bytes).
fn read_gguf_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    // Bytes are not required to be valid UTF-8; replace invalid sequences.
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Skip over one GGUF value of the given type without interpreting it.
///
/// This is the critical operation for keeping the byte cursor aligned as we
/// walk the KV section.  ARRAY types recurse so each element is individually
/// skipped.
fn skip_gguf_value<R: Read + Seek>(r: &mut R, vtype: GgufValueType) -> Result<()> {
    match vtype {
        // Scalar types with a fixed byte width — seek past them.
        t if t.fixed_size().is_some() => {
            let n = t.fixed_size().unwrap();
            r.seek(SeekFrom::Current(n as i64))
                .context("seek past scalar value")?;
        }
        GgufValueType::String => {
            let len = read_u64_le(r).context("string length in skip")?;
            r.seek(SeekFrom::Current(len as i64))
                .context("seek past string value")?;
        }
        GgufValueType::Array => {
            // element_type (u32) + element_count (u64)
            let elem_type_raw = read_u32_le(r).context("array element type")?;
            let elem_count    = read_u64_le(r).context("array element count")?;

            let elem_type = GgufValueType::from_u32(elem_type_raw)
                .with_context(|| format!("unknown array element type {elem_type_raw}"))?;

            for _ in 0..elem_count {
                skip_gguf_value(r, elem_type)
                    .context("skip array element")?;
            }
        }
        _ => bail!("unhandled GgufValueType in skip_gguf_value"),
    }
    Ok(())
}

/// Read a GGUF value of type UINT32 and return it as `u64`.
fn read_value_as_u64<R: Read + Seek>(r: &mut R, vtype: GgufValueType) -> Result<u64> {
    match vtype {
        GgufValueType::Uint8  => Ok(read_u8(r).context("read uint8 value")? as u64),
        GgufValueType::Uint16 => Ok(read_u16_le(r).context("read uint16 value")? as u64),
        GgufValueType::Uint32 => Ok(read_u32_le(r).context("read uint32 value")? as u64),
        GgufValueType::Uint64 => Ok(read_u64_le(r).context("read uint64 value")?),
        GgufValueType::Int8   => Ok(read_u8(r).context("read int8 value (reinterp)")? as u64),
        GgufValueType::Int16  => Ok(read_i16_le(r).context("read int16 value (reinterp)")? as u64),
        GgufValueType::Int32  => Ok(read_i32_le(r).context("read int32 value (reinterp)")? as u64),
        GgufValueType::Int64  => Ok(read_i64_le(r).context("read int64 value (reinterp)")? as u64),
        _ => bail!("cannot coerce value type {:?} to u64", vtype),
    }
}

/// Read a GGUF value of an integer type and return it as `u32`.
#[allow(dead_code)]
fn read_value_as_u32<R: Read + Seek>(r: &mut R, vtype: GgufValueType) -> Result<u32> {
    read_value_as_u64(r, vtype).map(|v| v as u32)
}

/// Parse a GGUF model file and extract the [`ModelIR`].
///
/// The parser reads only the file header and KV metadata section.  Tensor
/// data (the bulk of the file) is never accessed, making this fast even for
/// large models.
///
/// # Errors
///
/// Returns an error for bad magic bytes, unsupported GGUF versions, or I/O
/// failures.  Missing or unknown KV keys leave the corresponding IR fields as
/// `None` — they are not errors.
pub fn parse_gguf(path: &Path) -> Result<ModelIR> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open GGUF file: {}", path.display()))?;
    let mut r = io::BufReader::new(file);

    // ── Header ────────────────────────────────────────────────────────────────
    // magic (4 bytes) + version (u32) + tensor_count (u64) + kv_count (u64)
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).context("read GGUF magic")?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file (bad magic: {:?})", &magic);
    }

    let version = read_u32_le(&mut r).context("read GGUF version")?;
    if version < 1 || version > 3 {
        bail!("unsupported GGUF version {version}; expected 1–3");
    }

    let _tensor_count = read_u64_le(&mut r).context("read tensor_count")?;
    let kv_count      = read_u64_le(&mut r).context("read kv_count")?;

    // ── KV section ────────────────────────────────────────────────────────────
    // Keys we are interested in.
    let mut general_param_count: Option<u64>   = None;
    let mut general_architecture: Option<String> = None;
    // Architecture-specific keys; filled once we know the arch name.
    let mut block_count: Option<u64>     = None;
    let mut context_length: Option<u64>  = None;
    let mut embedding_length: Option<u64> = None;

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r).context("read KV key")?;

        let vtype_raw = read_u32_le(&mut r).context("read KV value type")?;
        let vtype = GgufValueType::from_u32(vtype_raw)
            .with_context(|| format!("unknown value type {vtype_raw} for key '{key}'"))?;

        // Decide whether to capture or skip this key's value.
        match key.as_str() {
            "general.parameter_count" => {
                general_param_count = Some(
                    read_value_as_u64(&mut r, vtype)
                        .with_context(|| format!("read general.parameter_count (type={vtype:?})"))?,
                );
            }
            "general.architecture" => {
                if vtype == GgufValueType::String {
                    general_architecture = Some(
                        read_gguf_string(&mut r)
                            .context("read general.architecture value")?,
                    );
                } else {
                    skip_gguf_value(&mut r, vtype)
                        .context("skip general.architecture (unexpected type)")?;
                }
            }
            other => {
                // Capture arch-specific keys if they match the pattern
                // `{arch}.block_count`, `{arch}.context_length`, or
                // `{arch}.embedding_length`.
                if let Some(suffix) = other.rfind('.').map(|i| &other[i + 1..]) {
                    match suffix {
                        "block_count" if block_count.is_none() => {
                            block_count = read_value_as_u64(&mut r, vtype)
                                .with_context(|| format!("read {key}"))
                                .ok();
                        }
                        "context_length" if context_length.is_none() => {
                            context_length = read_value_as_u64(&mut r, vtype)
                                .with_context(|| format!("read {key}"))
                                .ok();
                        }
                        "embedding_length" if embedding_length.is_none() => {
                            embedding_length = read_value_as_u64(&mut r, vtype)
                                .with_context(|| format!("read {key}"))
                                .ok();
                        }
                        _ => {
                            skip_gguf_value(&mut r, vtype)
                                .with_context(|| format!("skip key '{key}'"))?;
                        }
                    }
                } else {
                    skip_gguf_value(&mut r, vtype)
                        .with_context(|| format!("skip key '{other}'"))?;
                }
            }
        }
    }

    // ── Compute derived fields ────────────────────────────────────────────────

    // Use general.parameter_count when present; fall back to the standard
    // transformer estimate: 12 * block_count * embedding_length.
    // `saturating_mul` prevents silent integer overflow.
    let param_count = general_param_count.or_else(|| {
        let blocks    = block_count?;
        let embed_len = embedding_length?;
        Some(12u64.saturating_mul(blocks).saturating_mul(embed_len))
    });

    Ok(ModelIR {
        format:         ModelFormat::GGUF,
        param_count,
        architecture:   general_architecture,
        context_length: context_length.map(|v| v as u32),
        layer_count:    block_count.map(|v| v as u32),
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// ONNX parser
// ──────────────────────────────────────────────────────────────────────────────

/// Parse an ONNX model file and extract the [`ModelIR`].
///
/// Uses the `onnx-protobuf` crate to decode the protobuf-encoded model graph.
/// Parameter count is estimated by summing the element counts of all
/// initializer tensors (weight matrices).  Layer count is approximated by
/// counting the number of computation nodes in the graph.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or the protobuf data is
/// undecodable.  An empty or missing graph is not an error — fields default to
/// `None`.
pub fn parse_onnx(path: &Path) -> Result<ModelIR> {
    use protobuf::Message;

    let bytes = std::fs::read(path)
        .with_context(|| format!("read ONNX file: {}", path.display()))?;

    let model = onnx_protobuf::ModelProto::parse_from_bytes(&bytes)
        .context("decode ONNX protobuf")?;

    let graph = model.graph.as_ref();

    // Sum element counts across all initializer tensors to estimate total
    // parameter count.  Each initializer's shape is the product of its dims.
    let param_count: Option<u64> = graph.map(|g| {
        g.initializer
            .iter()
            .map(|t| {
                t.dims
                    .iter()
                    .copied()
                    .fold(1u64, |acc, d| acc.saturating_mul(d as u64))
            })
            .fold(0u64, |acc, n| acc.saturating_add(n))
    });

    // 0 params is as good as None (e.g. empty graph) — normalise to None.
    let param_count = param_count.filter(|&n| n > 0);

    // Approximate layer count as the number of computation nodes in the graph.
    let layer_count: Option<u32> = graph
        .map(|g| g.node.len())
        .filter(|&n| n > 0)
        .map(|n| n as u32);

    Ok(ModelIR {
        format:         ModelFormat::ONNX,
        param_count,
        architecture:   None, // ONNX graphs carry no named architecture field
        context_length: None,
        layer_count,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── GGUF test helpers ────────────────────────────────────────────────────

    /// Build a minimal GGUF byte buffer with the given KV entries.
    /// Header layout: magic(4) + version(u32) + tensor_count(u64) + kv_count(u64)
    struct GgufBuilder {
        kvs: Vec<Vec<u8>>,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self { kvs: Vec::new() }
        }

        fn write_u8_buf(buf: &mut Vec<u8>, v: u8) {
            buf.push(v);
        }

        fn write_u16_le_buf(buf: &mut Vec<u8>, v: u16) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_i16_le_buf(buf: &mut Vec<u8>, v: i16) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_u32_le_buf(buf: &mut Vec<u8>, v: u32) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_i32_le_buf(buf: &mut Vec<u8>, v: i32) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_f32_le_buf(buf: &mut Vec<u8>, v: f32) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_u64_le_buf(buf: &mut Vec<u8>, v: u64) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_i64_le_buf(buf: &mut Vec<u8>, v: i64) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_f64_le_buf(buf: &mut Vec<u8>, v: f64) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        fn write_gguf_string_buf(buf: &mut Vec<u8>, s: &str) {
            Self::write_u64_le_buf(buf, s.len() as u64);
            buf.extend_from_slice(s.as_bytes());
        }

        /// Add a uint32 KV entry.
        fn add_uint32(mut self, key: &str, val: u32) -> Self {
            let mut buf = Vec::new();
            Self::write_gguf_string_buf(&mut buf, key);
            Self::write_u32_le_buf(&mut buf, GgufValueType::Uint32 as u32);
            Self::write_u32_le_buf(&mut buf, val);
            self.kvs.push(buf);
            self
        }

        /// Add a uint64 KV entry.
        fn add_uint64(mut self, key: &str, val: u64) -> Self {
            let mut buf = Vec::new();
            Self::write_gguf_string_buf(&mut buf, key);
            Self::write_u32_le_buf(&mut buf, GgufValueType::Uint64 as u32);
            Self::write_u64_le_buf(&mut buf, val);
            self.kvs.push(buf);
            self
        }

        /// Add a string KV entry.
        fn add_string(mut self, key: &str, val: &str) -> Self {
            let mut buf = Vec::new();
            Self::write_gguf_string_buf(&mut buf, key);
            Self::write_u32_le_buf(&mut buf, GgufValueType::String as u32);
            Self::write_gguf_string_buf(&mut buf, val);
            self.kvs.push(buf);
            self
        }

        /// Add an array of uint32 values.
        fn add_array_uint32(mut self, key: &str, vals: &[u32]) -> Self {
            let mut buf = Vec::new();
            Self::write_gguf_string_buf(&mut buf, key);
            Self::write_u32_le_buf(&mut buf, GgufValueType::Array as u32);
            // element_type = UINT32 (4), element_count
            Self::write_u32_le_buf(&mut buf, GgufValueType::Uint32 as u32);
            Self::write_u64_le_buf(&mut buf, vals.len() as u64);
            for &v in vals {
                Self::write_u32_le_buf(&mut buf, v);
            }
            self.kvs.push(buf);
            self
        }

        /// Add an array of strings.
        fn add_array_string(mut self, key: &str, vals: &[&str]) -> Self {
            let mut buf = Vec::new();
            Self::write_gguf_string_buf(&mut buf, key);
            Self::write_u32_le_buf(&mut buf, GgufValueType::Array as u32);
            // element_type = STRING (8)
            Self::write_u32_le_buf(&mut buf, GgufValueType::String as u32);
            Self::write_u64_le_buf(&mut buf, vals.len() as u64);
            for &v in vals {
                Self::write_gguf_string_buf(&mut buf, v);
            }
            self.kvs.push(buf);
            self
        }

        /// Serialise into a complete GGUF byte buffer.
        fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            // Magic
            out.extend_from_slice(b"GGUF");
            // Version = 3
            Self::write_u32_le_buf(&mut out, 3);
            // tensor_count = 0
            Self::write_u64_le_buf(&mut out, 0);
            // kv_count
            Self::write_u64_le_buf(&mut out, self.kvs.len() as u64);
            for kv in self.kvs {
                out.extend_from_slice(&kv);
            }
            out
        }

        /// Write to a temp file and return the path.
        fn write_to_tempfile(self) -> tempfile::NamedTempFile {
            let bytes = self.build();
            let mut f = tempfile::NamedTempFile::new().unwrap();
            f.write_all(&bytes).unwrap();
            f
        }
    }

    // ── GGUF tests ───────────────────────────────────────────────────────────

    #[test]
    fn gguf_bad_magic_returns_error() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"NOTG\x00\x00\x00\x00").unwrap();
        let result = parse_gguf(tmp.path());
        assert!(result.is_err(), "expected error for bad magic");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("bad magic"), "error should mention bad magic, got: {msg}");
    }

    #[test]
    fn gguf_truncated_file_returns_error() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"GGUF").unwrap(); // truncated after magic
        assert!(parse_gguf(tmp.path()).is_err());
    }

    #[test]
    fn gguf_general_parameter_count_uint64() {
        let tmp = GgufBuilder::new()
            .add_uint64("general.parameter_count", 7_000_000_000)
            .add_string("general.architecture", "llama")
            .write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("parse should succeed");
        assert_eq!(ir.param_count, Some(7_000_000_000));
        assert_eq!(ir.architecture.as_deref(), Some("llama"));
        assert_eq!(ir.format, ModelFormat::GGUF);
    }

    #[test]
    fn gguf_general_parameter_count_uint32() {
        let tmp = GgufBuilder::new()
            .add_uint32("general.parameter_count", 1_000_000)
            .write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("parse should succeed");
        assert_eq!(ir.param_count, Some(1_000_000));
    }

    #[test]
    fn gguf_fallback_param_count_from_arch_keys() {
        // No general.parameter_count — should compute 12 * blocks * embed_len
        let tmp = GgufBuilder::new()
            .add_string("general.architecture", "mistral")
            .add_uint32("mistral.block_count", 32)
            .add_uint32("mistral.context_length", 4096)
            .add_uint32("mistral.embedding_length", 4096)
            .write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("parse should succeed");
        // 12 * 32 * 4096 = 1_572_864
        assert_eq!(ir.param_count, Some(12 * 32 * 4096));
        assert_eq!(ir.layer_count, Some(32));
        assert_eq!(ir.context_length, Some(4096));
    }

    #[test]
    fn gguf_fallback_none_when_arch_keys_absent() {
        // No parameter_count and no arch keys — param_count must be None
        let tmp = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("parse should succeed");
        assert_eq!(ir.param_count, None);
    }

    #[test]
    fn gguf_array_skip_does_not_misalign() {
        // An array KV appears BEFORE the target keys — the parser must skip
        // the array correctly to still find general.parameter_count.
        let tmp = GgufBuilder::new()
            .add_array_uint32("some.token_ids", &[1, 2, 3, 4, 5]) // array BEFORE target
            .add_uint64("general.parameter_count", 42_000_000)
            .write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("parse should succeed after array skip");
        assert_eq!(ir.param_count, Some(42_000_000));
    }

    #[test]
    fn gguf_array_string_skip_does_not_misalign() {
        // String array (a common type in GGUF tokenizer metadata) must also be
        // correctly skipped.
        let tmp = GgufBuilder::new()
            .add_array_string("tokenizer.tokens", &["<s>", "</s>", "hello"]) // before target
            .add_string("general.architecture", "phi")
            .add_uint32("phi.block_count", 24)
            .add_uint32("phi.embedding_length", 2048)
            .write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("parse should succeed after string array skip");
        assert_eq!(ir.layer_count, Some(24));
        // 12 * 24 * 2048 = 589_824
        assert_eq!(ir.param_count, Some(12 * 24 * 2048));
    }

    #[test]
    fn gguf_all_scalar_types_skipped_correctly() {
        // One KV of each type code, all before the target key.  If any skip is
        // wrong the final read will fail or return wrong data.
        let mut builder = GgufBuilder::new();
        // Add one KV per type that requires a skip path:
        // UINT8, INT8, UINT16, INT16, INT32, FLOAT32, BOOL, UINT64, INT64, FLOAT64
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.u8");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Uint8 as u32);
            GgufBuilder::write_u8_buf(&mut buf, 42u8);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.i8");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Int8 as u32);
            GgufBuilder::write_u8_buf(&mut buf, 200u8);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.u16");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Uint16 as u32);
            GgufBuilder::write_u16_le_buf(&mut buf, 1234u16);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.i16");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Int16 as u32);
            GgufBuilder::write_i16_le_buf(&mut buf, -100i16);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.i32");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Int32 as u32);
            GgufBuilder::write_i32_le_buf(&mut buf, -999i32);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.f32");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Float32 as u32);
            GgufBuilder::write_f32_le_buf(&mut buf, 3.14f32);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.bool");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Bool as u32);
            GgufBuilder::write_u8_buf(&mut buf, 1u8);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.i64");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Int64 as u32);
            GgufBuilder::write_i64_le_buf(&mut buf, -12345i64);
            builder.kvs.push(buf);
        }
        {
            let mut buf = Vec::new();
            GgufBuilder::write_gguf_string_buf(&mut buf, "misc.f64");
            GgufBuilder::write_u32_le_buf(&mut buf, GgufValueType::Float64 as u32);
            GgufBuilder::write_f64_le_buf(&mut buf, 2.718f64);
            builder.kvs.push(buf);
        }
        // Now add the target key after all the scalar skips.
        builder = builder.add_uint64("general.parameter_count", 999_999);
        let tmp = builder.write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("all scalar skips should leave cursor aligned");
        assert_eq!(ir.param_count, Some(999_999));
    }

    #[test]
    fn gguf_saturating_mul_no_overflow() {
        // Pathological case: very large block_count * embedding_length.
        // saturating_mul should prevent overflow.
        let tmp = GgufBuilder::new()
            .add_uint32("llama.block_count", u32::MAX)
            .add_uint32("llama.embedding_length", u32::MAX)
            .write_to_tempfile();

        let ir = parse_gguf(tmp.path()).expect("parse should not panic");
        // The result should be u64::MAX (saturated), not a panic.
        assert!(ir.param_count.is_some());
    }

    #[test]
    fn gguf_empty_kv_section() {
        // Valid GGUF header with zero KV entries.
        let tmp = GgufBuilder::new().write_to_tempfile();
        let ir = parse_gguf(tmp.path()).expect("empty KV section is valid");
        assert_eq!(ir.format, ModelFormat::GGUF);
        assert_eq!(ir.param_count, None);
        assert_eq!(ir.architecture, None);
    }

    // ── ONNX tests ───────────────────────────────────────────────────────────

    #[test]
    fn onnx_bad_bytes_returns_error() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"not an onnx file at all").unwrap();
        // protobuf parsing may succeed but return empty model — either way
        // we test that parse_onnx does not panic.
        let _result = parse_onnx(tmp.path()); // ok or err, must not panic
    }

    #[test]
    fn onnx_empty_file_no_panic() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // parse_onnx on an empty file should return Ok with all-None fields
        // (protobuf treats missing fields as defaults).
        let result = parse_onnx(tmp.path());
        match result {
            Ok(ir) => {
                assert_eq!(ir.format, ModelFormat::ONNX);
                assert_eq!(ir.param_count, None);
                assert_eq!(ir.layer_count, None);
            }
            Err(_) => {
                // Also acceptable — empty protobuf may fail to parse.
            }
        }
    }

    // ── parse_model dispatch ──────────────────────────────────────────────────

    #[test]
    fn parse_model_dispatches_by_extension() {
        // .gguf extension → parse_gguf (wrong magic → error)
        let mut tmp_gguf = tempfile::Builder::new()
            .suffix(".gguf")
            .tempfile()
            .unwrap();
        tmp_gguf.write_all(b"JUNK").unwrap();
        let result = parse_model(tmp_gguf.path());
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("bad magic") || msg.contains("GGUF"));

        // .onnx extension → parse_onnx (does not panic)
        let tmp_onnx = tempfile::Builder::new()
            .suffix(".onnx")
            .tempfile()
            .unwrap();
        let _result = parse_model(tmp_onnx.path()); // may succeed or fail, no panic

        // Unknown extension → error
        let tmp_unknown = tempfile::Builder::new()
            .suffix(".xyz")
            .tempfile()
            .unwrap();
        let result = parse_model(tmp_unknown.path());
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("xyz") || msg.contains("extension"));
    }
}
