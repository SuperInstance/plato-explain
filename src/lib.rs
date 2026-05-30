use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceType {
    TileValue,
    DeadbandRule,
    ModelPrediction,
    HistoricalPattern,
    FleetCorrelation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplanationFactor {
    pub source: String,
    pub contribution: f64,
    pub description: String,
    pub source_type: SourceType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step: usize,
    pub action: String,
    pub input_summary: String,
    pub output_summary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlternativePrediction {
    pub value: f64,
    pub confidence: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Explanation {
    pub prediction_id: Uuid,
    pub confidence: f64,
    pub factors: Vec<ExplanationFactor>,
    pub reasoning_chain: Vec<ReasoningStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplanationReport {
    pub prediction_id: Uuid,
    pub explanation: Explanation,
    pub alternatives: Vec<AlternativePrediction>,
    pub caveats: Vec<String>,
}

// ── Explanation impl ─────────────────────────────────────────────────────────

impl Explanation {
    pub fn new(prediction_id: Uuid) -> Self {
        Self {
            prediction_id,
            confidence: 0.0,
            factors: Vec::new(),
            reasoning_chain: Vec::new(),
        }
    }

    pub fn add_factor(&mut self, factor: ExplanationFactor) {
        self.factors.push(factor);
    }

    pub fn add_step(&mut self, step: ReasoningStep) {
        self.reasoning_chain.push(step);
    }

    /// Product of factor contributions as the aggregate confidence.
    pub fn total_confidence(&self) -> f64 {
        if self.factors.is_empty() {
            return 0.0;
        }
        self.factors.iter().map(|f| f.contribution).product()
    }

    pub fn top_factors(&self, k: usize) -> Vec<&ExplanationFactor> {
        let mut refs: Vec<&ExplanationFactor> = self.factors.iter().collect();
        refs.sort_by(|a, b| b.contribution.partial_cmp(&a.contribution).unwrap_or(std::cmp::Ordering::Equal));
        refs.truncate(k);
        refs
    }
}

// ── ExplanationReport impl ───────────────────────────────────────────────────

impl ExplanationReport {
    pub fn generate(prediction_id: Uuid, explanation: Explanation) -> Self {
        Self {
            prediction_id,
            explanation,
            alternatives: Vec::new(),
            caveats: Vec::new(),
        }
    }

    pub fn to_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Prediction Report: {}", self.prediction_id));
        lines.push(format!("Overall Confidence: {:.4}", self.explanation.confidence));
        lines.push(String::new());

        lines.push("Contributing Factors:".into());
        for (i, f) in self.explanation.factors.iter().enumerate() {
            lines.push(format!(
                "  {}. [{:?}] {} (contribution: {:.4}) — {}",
                i + 1,
                f.source_type,
                f.source,
                f.contribution,
                f.description
            ));
        }

        if !self.explanation.reasoning_chain.is_empty() {
            lines.push(String::new());
            lines.push("Reasoning Chain:".into());
            for s in &self.explanation.reasoning_chain {
                lines.push(format!(
                    "  Step {}: {} | in: {} | out: {}",
                    s.step, s.action, s.input_summary, s.output_summary
                ));
            }
        }

        if !self.alternatives.is_empty() {
            lines.push(String::new());
            lines.push("Alternative Predictions:".into());
            for a in &self.alternatives {
                lines.push(format!(
                    "  value={:.4} confidence={:.4} — {}",
                    a.value, a.confidence, a.reasoning
                ));
            }
        }

        if !self.caveats.is_empty() {
            lines.push(String::new());
            lines.push("Caveats:".into());
            for c in &self.caveats {
                lines.push(format!("  - {}", c));
            }
        }

        lines.join("\n")
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

// ── Constructor helpers ──────────────────────────────────────────────────────

pub fn deadband_explanation(value: f64, threshold: f64, resolved: bool) -> Explanation {
    let id = Uuid::new_v4();
    let mut ex = Explanation::new(id);
    ex.add_factor(ExplanationFactor {
        source: "deadband_check".into(),
        contribution: if resolved { 0.95 } else { 0.5 },
        description: format!(
            "Value {:.4} compared against threshold {:.4}; resolved={}",
            value, threshold, resolved
        ),
        source_type: SourceType::DeadbandRule,
    });
    ex.add_step(ReasoningStep {
        step: 1,
        action: "deadband_evaluation".into(),
        input_summary: format!("value={:.4}, threshold={:.4}", value, threshold),
        output_summary: format!("resolved={}", resolved),
    });
    ex.confidence = ex.total_confidence();
    ex
}

pub fn model_explanation(model_name: &str, input: &str, output: &str, confidence: f64) -> Explanation {
    let id = Uuid::new_v4();
    let mut ex = Explanation::new(id);
    ex.add_factor(ExplanationFactor {
        source: model_name.into(),
        contribution: confidence,
        description: format!("Model {} produced output '{}' from input '{}'", model_name, output, input),
        source_type: SourceType::ModelPrediction,
    });
    ex.add_step(ReasoningStep {
        step: 1,
        action: format!("run_model_{}", model_name),
        input_summary: input.into(),
        output_summary: output.into(),
    });
    ex.confidence = confidence;
    ex
}

pub fn chain_explanation(steps: &[ReasoningStep]) -> Explanation {
    let id = Uuid::new_v4();
    let mut ex = Explanation::new(id);
    for s in steps {
        ex.add_step(s.clone());
    }
    let n = steps.len() as f64;
    ex.add_factor(ExplanationFactor {
        source: "chain".into(),
        contribution: if n > 0.0 { 1.0 / n } else { 0.0 },
        description: format!("{}-step reasoning chain", steps.len()),
        source_type: SourceType::HistoricalPattern,
    });
    ex.confidence = ex.total_confidence();
    ex
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_factor(contribution: f64) -> ExplanationFactor {
        ExplanationFactor {
            source: "test_source".into(),
            contribution,
            description: "test factor".into(),
            source_type: SourceType::TileValue,
        }
    }

    fn sample_step(n: usize) -> ReasoningStep {
        ReasoningStep {
            step: n,
            action: format!("action_{}", n),
            input_summary: format!("in_{}", n),
            output_summary: format!("out_{}", n),
        }
    }

    // 1. Factor addition
    #[test]
    fn factor_addition() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        assert!(ex.factors.is_empty());
        ex.add_factor(sample_factor(0.8));
        assert_eq!(ex.factors.len(), 1);
        assert_eq!(ex.factors[0].contribution, 0.8);
    }

    // 2. Factor retrieval
    #[test]
    fn factor_retrieval() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_factor(ExplanationFactor {
            source: "sensor_a".into(),
            contribution: 0.9,
            description: "temp reading".into(),
            source_type: SourceType::TileValue,
        });
        assert_eq!(ex.factors[0].source, "sensor_a");
        assert_eq!(ex.factors[0].source_type, SourceType::TileValue);
    }

    // 3. Total confidence — multiple factors (product)
    #[test]
    fn total_confidence_product() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_factor(sample_factor(0.8));
        ex.add_factor(sample_factor(0.9));
        let expected = 0.8 * 0.9;
        assert!((ex.total_confidence() - expected).abs() < 1e-10);
    }

    // 4. Total confidence — empty
    #[test]
    fn total_confidence_empty() {
        let ex = Explanation::new(Uuid::new_v4());
        assert_eq!(ex.total_confidence(), 0.0);
    }

    // 5. Total confidence — zero factor
    #[test]
    fn total_confidence_zero() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_factor(sample_factor(0.0));
        ex.add_factor(sample_factor(0.9));
        assert_eq!(ex.total_confidence(), 0.0);
    }

    // 6. Total confidence — perfect
    #[test]
    fn total_confidence_perfect() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_factor(sample_factor(1.0));
        assert_eq!(ex.total_confidence(), 1.0);
    }

    // 7. Total confidence — mixed
    #[test]
    fn total_confidence_mixed() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_factor(sample_factor(0.5));
        ex.add_factor(sample_factor(0.6));
        ex.add_factor(sample_factor(0.7));
        let expected = 0.5 * 0.6 * 0.7;
        assert!((ex.total_confidence() - expected).abs() < 1e-10);
    }

    // 8. Top-k factors
    #[test]
    fn top_k_factors() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_factor(sample_factor(0.5));
        ex.add_factor(sample_factor(0.9));
        ex.add_factor(sample_factor(0.7));
        let top = ex.top_factors(2);
        assert_eq!(top.len(), 2);
        assert!((top[0].contribution - 0.9).abs() < 1e-10);
        assert!((top[1].contribution - 0.7).abs() < 1e-10);
    }

    // 9. Reasoning chain construction
    #[test]
    fn reasoning_chain_construction() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_step(sample_step(1));
        ex.add_step(sample_step(2));
        assert_eq!(ex.reasoning_chain.len(), 2);
        assert_eq!(ex.reasoning_chain[0].action, "action_1");
        assert_eq!(ex.reasoning_chain[1].step, 2);
    }

    // 10. Deadband explanation
    #[test]
    fn deadband_explanation_test() {
        let ex = deadband_explanation(1.5, 2.0, true);
        assert_eq!(ex.factors.len(), 1);
        assert_eq!(ex.factors[0].source_type, SourceType::DeadbandRule);
        assert_eq!(ex.reasoning_chain.len(), 1);
        assert!((ex.confidence - 0.95).abs() < 1e-10);
    }

    // 11. Deadband explanation — unresolved
    #[test]
    fn deadband_explanation_unresolved() {
        let ex = deadband_explanation(3.0, 2.0, false);
        assert!((ex.factors[0].contribution - 0.5).abs() < 1e-10);
    }

    // 12. Model explanation
    #[test]
    fn model_explanation_test() {
        let ex = model_explanation("lstm_v2", "tile_data", "prediction", 0.88);
        assert_eq!(ex.factors[0].source, "lstm_v2");
        assert_eq!(ex.factors[0].source_type, SourceType::ModelPrediction);
        assert!((ex.confidence - 0.88).abs() < 1e-10);
    }

    // 13. Chain explanation
    #[test]
    fn chain_explanation_test() {
        let steps = vec![sample_step(1), sample_step(2), sample_step(3)];
        let ex = chain_explanation(&steps);
        assert_eq!(ex.reasoning_chain.len(), 3);
        assert_eq!(ex.factors.len(), 1);
    }

    // 14. Report text generation
    #[test]
    fn report_text_generation() {
        let id = Uuid::new_v4();
        let ex = deadband_explanation(1.0, 2.0, true);
        let report = ExplanationReport::generate(id, ex);
        let text = report.to_text();
        assert!(text.contains("Prediction Report"));
        assert!(text.contains("Contributing Factors"));
    }

    // 15. Report JSON generation
    #[test]
    fn report_json_generation() {
        let id = Uuid::new_v4();
        let ex = model_explanation("test_model", "in", "out", 0.9);
        let report = ExplanationReport::generate(id, ex);
        let json = report.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["prediction_id"], id.to_string());
    }

    // 16. Alternatives and caveats
    #[test]
    fn alternatives_and_caveats() {
        let id = Uuid::new_v4();
        let ex = Explanation::new(id);
        let mut report = ExplanationReport::generate(id, ex);
        report.alternatives.push(AlternativePrediction {
            value: 42.0,
            confidence: 0.7,
            reasoning: "fallback model".into(),
        });
        report.caveats.push("Low sensor coverage".into());
        let text = report.to_text();
        assert!(text.contains("Alternative Predictions"));
        assert!(text.contains("Caveats"));
        assert!(text.contains("Low sensor coverage"));
    }

    // 17. Empty explanation edge case
    #[test]
    fn empty_explanation() {
        let ex = Explanation::new(Uuid::new_v4());
        assert!(ex.factors.is_empty());
        assert!(ex.reasoning_chain.is_empty());
        assert_eq!(ex.total_confidence(), 0.0);
        assert!(ex.top_factors(5).is_empty());
    }

    // 18. Single factor
    #[test]
    fn single_factor() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        ex.add_factor(sample_factor(0.77));
        assert_eq!(ex.total_confidence(), 0.77);
        let top = ex.top_factors(3);
        assert_eq!(top.len(), 1);
    }

    // 19. Many factors
    #[test]
    fn many_factors() {
        let id = Uuid::new_v4();
        let mut ex = Explanation::new(id);
        for i in 0..100 {
            ex.add_factor(ExplanationFactor {
                source: format!("factor_{}", i),
                contribution: (i as f64) / 100.0,
                description: "many".into(),
                source_type: SourceType::FleetCorrelation,
            });
        }
        assert_eq!(ex.factors.len(), 100);
        let top = ex.top_factors(5);
        assert_eq!(top.len(), 5);
        // Top should be the last factors (highest contribution)
        assert!((top[0].contribution - 0.99).abs() < 1e-10);
    }

    // 20. Chain explanation with empty steps
    #[test]
    fn chain_explanation_empty() {
        let ex = chain_explanation(&[]);
        assert!(ex.reasoning_chain.is_empty());
        assert_eq!(ex.factors[0].contribution, 0.0);
    }
}
