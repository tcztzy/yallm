use std::{hint::black_box, time::Instant};

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

fn env_budget_ns(name: &str, default: u128) -> u128 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u128>().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "run in CI via --release for stable timing"]
fn perf_guardrail_openai_downstream_to_ir() {
    let req = sample_openai_request();
    let iters = 20_000_u128;
    let started = Instant::now();
    for _ in 0..iters {
        let ir = openai_downstream_to_ir(black_box(&req)).expect("valid request");
        black_box(ir);
    }
    let elapsed = started.elapsed().as_nanos();
    let ns_per_iter = elapsed / iters;
    let budget = env_budget_ns("YALLM_BUDGET_OPENAI_TO_IR_NS", 50_000);
    assert!(
        ns_per_iter <= budget,
        "openai_downstream_to_ir exceeded budget: {ns_per_iter} ns/iter > {budget} ns/iter"
    );
}

#[test]
#[ignore = "run in CI via --release for stable timing"]
fn perf_guardrail_ir_to_openai_response() {
    let req = sample_openai_request();
    let ir = openai_downstream_to_ir(&req).expect("valid request");
    let resp = ChatResponse {
        id: "perf_resp".to_string(),
        model: "gpt-4o-mini".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message::text(Role::Assistant, "Hello"),
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(Usage {
            prompt_tokens: 8,
            completion_tokens: 4,
            total_tokens: 12,
        }),
    };

    let iters = 20_000_u128;
    let started = Instant::now();
    for _ in 0..iters {
        let out = ir_to_openai_downstream_response(black_box(&resp), black_box(&ir.model));
        black_box(out);
    }
    let elapsed = started.elapsed().as_nanos();
    let ns_per_iter = elapsed / iters;
    let budget = env_budget_ns("YALLM_BUDGET_IR_TO_OPENAI_NS", 80_000);
    assert!(
        ns_per_iter <= budget,
        "ir_to_openai_downstream_response exceeded budget: {ns_per_iter} ns/iter > {budget} ns/iter"
    );
}
