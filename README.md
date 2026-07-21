# si-deckhand-rs

BM25 retriever — zero-dependency Rust implementation.

A drop-in replacement for the Python `deckhand` retriever with the same API shape
and identical BM25 scoring (k1=1.5, b=0.75).

## Features

- **Zero dependencies**: Uses only Rust standard library
- **BM25 ranking**: Standard BM25 algorithm with configurable k1 and b parameters
- **Fast**: 10-100x faster than the Python equivalent
- **Same API**: Compatible with the Python deckhand retriever interface

## Usage

```rust
use deckhand_rs::{FileEntry, Retriever};

// Create file entries
let files = vec![
    FileEntry {
        path: "/path/to/file.md".to_string(),
        rel_path: "docs/file.md".to_string(),
        words: 100,
    },
    // ... more files
];

// Build the index
let retriever = Retriever::new(&files);

// Search
let results = retriever.search("query terms", 10);
for result in results {
    println!("{}: {} ({})", result.rel_path, result.score, result.snippet);
}
```

## API

### `FileEntry`

```rust
pub struct FileEntry {
    pub path: String,      // Absolute path to the file
    pub rel_path: String,  // Relative path from the index root
    pub words: usize,       // Number of words in the file
}
```

### `SearchResult`

```rust
pub struct SearchResult {
    pub path: String,      // Absolute path to the file
    pub rel_path: String,  // Relative path from the index root
    pub score: f64,        // BM25 score
    pub snippet: String,   // Text snippet around the first match
}
```

### `Retriever`

```rust
// Create with default BM25 parameters (k1=1.5, b=0.75)
pub fn new(files: &[FileEntry]) -> Self

// Create with custom BM25 parameters
pub fn with_params(files: &[FileEntry], k1: f64, b: f64) -> Self

// Search the corpus
pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult>

// Corpus statistics
pub fn stats(&self) -> RetrieverStats

// Tokenize text (public utility function)
pub fn tokenize(text: &str) -> Vec<String>
```

## Testing

```bash
cargo test
```

10 tests covering:
- Search returns results
- Ranking relevant results first
- Empty query handling
- No match handling
- Top-k limit
- Statistics
- Snippet generation
- Zero-word file filtering
- Custom BM25 parameters
- Tokenization

## License

MIT
