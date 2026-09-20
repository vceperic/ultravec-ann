use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn write_fvecs(path: &Path, rows: usize, dim: usize) {
    let mut file = File::create(path).unwrap();
    for row in 0..rows {
        file.write_all(&(dim as i32).to_le_bytes()).unwrap();
        for column in 0..dim {
            let value = (((row * 17 + column * 13) % 97) as f32 + 1.0) / 97.0;
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
}

#[test]
fn external_query_header_reports_loaded_query_count() {
    let unique = format!("ultravec-cli-test-{}", std::process::id());
    let temporary = std::env::temp_dir().join(&unique);
    std::fs::create_dir_all(&temporary).unwrap();
    let base = temporary.join(format!("{unique}-base.fvecs"));
    let queries = temporary.join(format!("{unique}-query.fvecs"));
    write_fvecs(&base, 128, 8);
    write_fvecs(&queries, 3, 8);

    let output = Command::new(env!("CARGO_BIN_EXE_ultravec"))
        .args([
            "bench",
            "--dataset",
            base.to_str().unwrap(),
            "--query-file",
            queries.to_str().unwrap(),
            "--query-max",
            "2",
            "--bits",
            "2",
            "--seed",
            "42",
            "--lean",
        ])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "benchmark failed:\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("external queries: 2"));
    assert!(stdout.contains("evaluated queries: 2"));
    assert!(!stdout.contains("queries 150"));

    let results = Path::new(env!("CARGO_MANIFEST_DIR")).join("results");
    let dataset = format!("{unique}-base");
    let _ = std::fs::remove_file(results.join(format!("flat-{dataset}.md")));
    let _ = std::fs::remove_file(results.join(format!("flat-{dataset}-per-query.csv")));
    std::fs::remove_dir_all(temporary).unwrap();
}
