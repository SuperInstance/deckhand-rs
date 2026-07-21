//! BM25 retriever — zero-dependency Rust implementation.
//!
//! Drop-in replacement for the Python deckhand retriever with the same API shape
//! and identical BM25 scoring (k1=1.5, b=0.75).

use std::collections::{HashMap, HashSet};
use std::fs;

/// A single file entry in the index.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Absolute path to the file.
    pub path: String,
    /// Relative path from the index root.
    pub rel_path: String,
    /// Number of words in the file (estimated).
    pub words: usize,
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Absolute path to the file.
    pub path: String,
    /// Relative path from the index root.
    pub rel_path: String,
    /// BM25 score.
    pub score: f64,
    /// Text snippet around the first match.
    pub snippet: String,
}

/// BM25 search over indexed files.
#[derive(Debug)]
pub struct Retriever {
    /// BM25 k1 parameter (default: 1.5).
    k1: f64,
    /// BM25 b parameter (default: 0.75).
    b: f64,
    /// Files in the index (filtered to those with words > 0).
    files: Vec<FileEntry>,
    /// Total number of documents.
    total_docs: usize,
    /// Average document length.
    avg_len: f64,
    /// Document lengths.
    doc_len: Vec<usize>,
    /// Term frequencies per document: term -> count.
    term_freqs: Vec<HashMap<String, usize>>,
    /// Document frequency: term -> number of docs containing term.
    doc_freq: HashMap<String, usize>,
    /// Tokenized documents for snippet generation.
    doc_tokens: Vec<Vec<String>>,
}

impl Retriever {
    /// Create a new retriever with default BM25 parameters (k1=1.5, b=0.75).
    pub fn new(files: &[FileEntry]) -> Self {
        Self::with_params(files, 1.5, 0.75)
    }

    /// Create a new retriever with custom BM25 parameters.
    pub fn with_params(files: &[FileEntry], k1: f64, b: f64) -> Self {
        let files: Vec<FileEntry> = files
            .iter()
            .filter(|f| f.words > 0)
            .cloned()
            .collect();

        let total_docs = files.len();
        let mut retriever = Retriever {
            k1,
            b,
            files,
            total_docs,
            avg_len: 0.0,
            doc_len: Vec::new(),
            term_freqs: Vec::new(),
            doc_freq: HashMap::new(),
            doc_tokens: Vec::new(),
        };

        retriever.build_index();
        retriever
    }

    /// Tokenize text into lowercase alphanumeric words.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Build the BM25 index from file entries.
    pub fn build_index(&mut self) {
        let mut total_tokens: usize = 0;
        let mut doc_freq_seen: HashSet<String> = HashSet::new();

        for f in &self.files {
            // Read file content
            let content = self.read_file(&f.path);

            // Tokenize
            let tokens = Self::tokenize(&content);
            self.doc_tokens.push(tokens.clone());

            // Count term frequencies
            let mut tf: HashMap<String, usize> = HashMap::new();
            for token in &tokens {
                *tf.entry(token.clone()).or_insert(0) += 1;
            }
            self.term_freqs.push(tf);

            // Update document frequencies
            doc_freq_seen.clear();
            for token in &tokens {
                if doc_freq_seen.insert(token.clone()) {
                    *self.doc_freq.entry(token.clone()).or_insert(0) += 1;
                }
            }

            let len = tokens.len();
            self.doc_len.push(len);
            total_tokens += len;
        }

        self.avg_len = if self.total_docs > 0 {
            total_tokens as f64 / self.total_docs as f64
        } else {
            1.0
        };
    }

    /// Read file content, ignoring errors.
    fn read_file(&self, path: &str) -> String {
        fs::read_to_string(path).unwrap_or_default()
    }

    /// Compute IDF for a term.
    fn idf(&self, term: &str) -> f64 {
        let df = *self.doc_freq.get(term).unwrap_or(&0);
        if df == 0 {
            return 0.0;
        }
        let n = self.total_docs as f64;
        ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln()
    }

    /// Compute BM25 score for a query against a document.
    fn bm25_score(&self, query_terms: &[String], doc_idx: usize) -> f64 {
        let mut score = 0.0;
        let tf = &self.term_freqs[doc_idx];
        let dl = self.doc_len[doc_idx] as f64;
        let avg_len = self.avg_len.max(1.0);

        for term in query_terms {
            let idf = self.idf(term);
            if idf == 0.0 {
                continue;
            }
            let f = *tf.get(term).unwrap_or(&0);
            if f == 0 {
                continue;
            }
            let f = f as f64;
            let numerator = f * (self.k1 + 1.0);
            let denominator = f + self.k1 * (1.0 - self.b + self.b * dl / avg_len);
            score += idf * numerator / denominator;
        }

        score
    }

    /// Search the corpus. Returns ranked results.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f64)> = Vec::new();
        for i in 0..self.total_docs {
            let s = self.bm25_score(&query_terms, i);
            if s > 0.0 {
                scored.push((i, s));
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut results = Vec::new();
        for (idx, score) in scored.into_iter().take(top_k) {
            let f = &self.files[idx];
            let snippet = self.snippet(idx, &query_terms, 50);
            results.push(SearchResult {
                path: f.path.clone(),
                rel_path: f.rel_path.clone(),
                score,
                snippet,
            });
        }

        results
    }

    /// Extract a snippet around the first query term match.
    fn snippet(&self, doc_idx: usize, query_terms: &[String], context: usize) -> String {
        let tokens = &self.doc_tokens[doc_idx];
        let query_set: HashSet<&str> = query_terms.iter().map(|s| s.as_str()).collect();

        for (i, tok) in tokens.iter().enumerate() {
            if query_set.contains(tok.as_str()) {
                let start = if i > context { i - context } else { 0 };
                let end = (i + context).min(tokens.len());
                let snippet_tokens = &tokens[start..end];
                return format!("...{}...", snippet_tokens.join(" "));
            }
        }

        String::new()
    }

    /// Corpus statistics.
    pub fn stats(&self) -> RetrieverStats {
        RetrieverStats {
            total_docs: self.total_docs,
            avg_doc_length: self.avg_len * 10.0_f64.round() / 10.0,
            total_terms: self.doc_len.iter().sum(),
            unique_terms: self.doc_freq.len(),
        }
    }
}

/// Statistics about the indexed corpus.
#[derive(Debug)]
pub struct RetrieverStats {
    pub total_docs: usize,
    pub avg_doc_length: f64,
    pub total_terms: usize,
    pub unique_terms: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn test_temp_dir() -> String {
        format!("/tmp/test_deckhand_{}_{}", std::process::id(), TEST_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    fn test_create_file(path: &str, content: &str) -> FileEntry {
        let full_path = Path::new(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();

        let word_count = content.split_whitespace().count();
        FileEntry {
            path: path.to_string(),
            rel_path: path.replace("/tmp/test_deckhand_", "").splitn(2, '/').last().unwrap_or(path).to_string(),
            words: word_count,
        }
    }

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn test_temp_dir() -> String {
        // Use atomic counter to create unique test directory per test
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("/tmp/test_deckhand_{}_{}", std::process::id(), id)
    }

    #[test]
    fn test_search_returns_results() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/conservation.md", dir),
                "The conservation law of intelligence states that every mind runs on a budget."),
            test_create_file(&format!("{}/docs/charts.md", dir),
                "Languages are navigation charts of the same territory drawn for different readers."),
            test_create_file(&format!("{}/docs/boat.md", dir),
                "The boat is the reference implementation for edge-first thinking at 12 volts."),
            test_create_file(&format!("{}/docs/rate_limit.md", dir),
                "The rate limit is a conservation fence that prevents overconsumption of tokens."),
        ];

        let retriever = Retriever::new(&files);
        let results = retriever.search("conservation law budget", 10);

        assert!(!results.is_empty());
        assert!(results[0].rel_path.to_lowercase().contains("conservation"));

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_search_ranks_relevant_first() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/conservation.md", dir),
                "The conservation law of intelligence states that every mind runs on a budget."),
            test_create_file(&format!("{}/docs/charts.md", dir),
                "Languages are navigation charts of the same territory drawn for different readers."),
            test_create_file(&format!("{}/docs/boat.md", dir),
                "The boat is the reference implementation for edge-first thinking at 12 volts."),
            test_create_file(&format!("{}/docs/rate_limit.md", dir),
                "The rate limit is a conservation fence that prevents overconsumption of tokens."),
        ];

        let retriever = Retriever::new(&files);
        let results = retriever.search("boat edge-first", 10);

        assert!(!results.is_empty());
        assert!(results[0].rel_path.to_lowercase().contains("boat"));

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_empty_query_returns_nothing() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/test.md", dir), "Some test content here."),
        ];

        let retriever = Retriever::new(&files);
        let results = retriever.search("", 10);

        assert!(results.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_no_match_returns_empty() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/test.md", dir), "Some test content here."),
        ];

        let retriever = Retriever::new(&files);
        let results = retriever.search("quantum_entanglement_teleportation_xyzzy", 10);

        assert!(results.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_top_k_limit() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/conservation.md", dir),
                "The conservation law of intelligence states that every mind runs on a budget."),
            test_create_file(&format!("{}/docs/charts.md", dir),
                "Languages are navigation charts of the same territory drawn for different readers."),
            test_create_file(&format!("{}/docs/boat.md", dir),
                "The boat is the reference implementation for edge-first thinking at 12 volts."),
            test_create_file(&format!("{}/docs/rate_limit.md", dir),
                "The rate limit is a conservation fence that prevents overconsumption of tokens."),
        ];

        let retriever = Retriever::new(&files);
        let results = retriever.search("the", 2);

        assert!(results.len() <= 2);

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_stats() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/conservation.md", dir),
                "The conservation law of intelligence states that every mind runs on a budget."),
            test_create_file(&format!("{}/docs/charts.md", dir),
                "Languages are navigation charts of the same territory drawn for different readers."),
            test_create_file(&format!("{}/docs/boat.md", dir),
                "The boat is the reference implementation for edge-first thinking at 12 volts."),
            test_create_file(&format!("{}/docs/rate_limit.md", dir),
                "The rate limit is a conservation fence that prevents overconsumption of tokens."),
        ];

        let retriever = Retriever::new(&files);
        let stats = retriever.stats();

        assert_eq!(stats.total_docs, 4);
        assert!(stats.avg_doc_length > 0.0);
        assert!(stats.unique_terms > 10);

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_snippet_generation() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/conservation.md", dir),
                "The conservation law of intelligence states that every mind runs on a budget."),
            test_create_file(&format!("{}/docs/charts.md", dir),
                "Languages are navigation charts of the same territory drawn for different readers."),
            test_create_file(&format!("{}/docs/boat.md", dir),
                "The boat is the reference implementation for edge-first thinking at 12 volts."),
            test_create_file(&format!("{}/docs/rate_limit.md", dir),
                "The rate limit is a conservation fence that prevents overconsumption of tokens."),
        ];

        let retriever = Retriever::new(&files);
        let results = retriever.search("conservation", 10);

        assert!(!results.is_empty());
        assert!(!results[0].snippet.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_filters_zero_word_files() {
        let dir = test_temp_dir();
        let files = vec![
            FileEntry {
                path: format!("{}/empty.md", dir),
                rel_path: "empty.md".to_string(),
                words: 0,
            },
            test_create_file(&format!("{}/docs/content.md", dir), "This file has words."),
        ];

        let retriever = Retriever::new(&files);
        let stats = retriever.stats();

        // Only non-zero word files should be indexed
        assert_eq!(stats.total_docs, 1);

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_custom_bm25_params() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/test.md", dir), "test test test content"),
        ];

        let retriever = Retriever::with_params(&files, 2.0, 0.5);
        let results = retriever.search("test", 10);

        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_tokenization() {
        let dir = test_temp_dir();
        let files = vec![
            test_create_file(&format!("{}/docs/test.md", dir),
                "Hello, World! This is a TEST... with UPPERCASE and lowercase."),
        ];

        let retriever = Retriever::new(&files);
        let results = retriever.search("hello world test", 10);

        // All variations should match due to lowercase tokenization
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.0);

        // Cleanup
        let _ = fs::remove_dir_all(dir);
    }
}
