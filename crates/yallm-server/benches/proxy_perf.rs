use criterion::{Criterion, black_box, criterion_group, criterion_main};
use serde_json::json;
use yallm_ir::{ChatResponse, Choice, Message, Role, Usage};
use yallm_server::{ir_to_openai_downstream_response, openai_downstream_to_ir};

fn sample_openai_request() -> serde_json::Value {
    json!({
        "model": "openai:gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Explain quaternions simply."}
        ],
        "max_tokens": 128,
        "temperature": 0.2
    })
}

fn bench_openai_downstream_to_ir(c: &mut Criterion) {
    let req = sample_openai_request();
    c.bench_function("openai_downstream_to_ir", |b| {
        b.iter(|| {
            let ir = openai_downstream_to_ir(black_box(&req)).expect("valid request");
            black_box(ir);
        })
    });
}

fn bench_ir_to_openai_downstream_response(c: &mut Criterion) {
    let req = sample_openai_request();
    let ir = openai_downstream_to_ir(&req).expect("valid request");
    let mock = ChatResponse {
        id: "bench_resp".to_string(),
        model: "gpt-4o-mini".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message::text(Role::Assistant, "Hello bench"),
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(Usage {
            prompt_tokens: 8,
            completion_tokens: 4,
            total_tokens: 12,
        }),
    };

    c.bench_function("ir_to_openai_downstream_response", |b| {
        b.iter(|| {
            let out = ir_to_openai_downstream_response(black_box(&mock), black_box(&ir.model));
            black_box(out);
        })
    });
}

criterion_group!(
    name = proxy_perf;
    config = Criterion::default();
    targets = bench_openai_downstream_to_ir, bench_ir_to_openai_downstream_response
);
criterion_main!(proxy_perf);
