// =================================================================================
// BAKOME-CardioRisk-Pro – Prédiction de risque cardiovasculaire (Version Pro)
// Auteur : BAKOME (Kitoko Bakome Fabrice Bandia)
// Licence : MIT
// Description : Outil de prédiction médicale utilisant la régression logistique,
//              validation croisée, métriques avancées, interface web, export PDF.
// =================================================================================

use axum::{
    Router, routing::{get, post}, Json, extract::{State, Multipart, Query},
    response::{IntoResponse, Html, Redirect},
};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, collections::HashMap, error::Error, fs, io::Write, path::PathBuf};
use tokio::sync::Mutex;
use nalgebra as na;
use na::{DMatrix, DVector};
use csv::ReaderBuilder;
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tracing::{info, error, warn};
use tracing_subscriber;
use std::time::Instant;
use std::fmt;

// ------------------------------------------------------------
// 1. MODÈLE DE RÉGRESSION LOGISTIQUE AVEC OPTIMISATIONS
// ------------------------------------------------------------
#[derive(Clone, Debug)]
struct LogisticRegression {
    coefficients: DVector<f64>,
    intercept: f64,
    num_features: usize,
}

impl LogisticRegression {
    fn new(num_features: usize) -> Self {
        Self {
            coefficients: DVector::zeros(num_features),
            intercept: 0.0,
            num_features,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn predict_prob(&self, features: &DVector<f64>) -> f64 {
        let z = self.coefficients.dot(features) + self.intercept;
        Self::sigmoid(z)
    }

    fn predict_binary(&self, features: &DVector<f64>, threshold: f64) -> u8 {
        if self.predict_prob(features) >= threshold { 1 } else { 0 }
    }

    fn train(&mut self, x: &DMatrix<f64>, y: &DVector<f64>, learning_rate: f64, epochs: usize) {
        let mut grad_coef = DVector::zeros(self.num_features);
        let mut grad_intercept = 0.0;
        let n = x.nrows() as f64;
        for _ in 0..epochs {
            grad_coef.fill(0.0);
            grad_intercept = 0.0;
            for i in 0..x.nrows() {
                let row = x.row(i).transpose();
                let pred = self.predict_prob(&row);
                let error = pred - y[i];
                grad_coef += &(row * error);
                grad_intercept += error;
            }
            self.coefficients -= &(grad_coef * learning_rate / n);
            self.intercept -= grad_intercept * learning_rate / n;
        }
    }

    fn coefficients_vec(&self) -> Vec<f64> {
        self.coefficients.as_slice().to_vec()
    }

    fn save_binary(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let data = bincode::serialize(&(self.coefficients.as_slice(), self.intercept))?;
        fs::write(path, data)?;
        Ok(())
    }

    fn load_binary(path: &str, num_features: usize) -> Result<Self, Box<dyn Error>> {
        let data = fs::read(path)?;
        let (coef_slice, intercept): (Vec<f64>, f64) = bincode::deserialize(&data)?;
        let mut coefficients = DVector::zeros(num_features);
        for (i, &v) in coef_slice.iter().enumerate() {
            coefficients[i] = v;
        }
        Ok(Self { coefficients, intercept, num_features })
    }
}

// ------------------------------------------------------------
// 2. STRUCTURE DE DONNÉES PATIENT
// ------------------------------------------------------------
#[derive(Debug, Deserialize, Serialize, Clone)]
struct PatientRecord {
    age: f64,
    sys_bp: f64,
    tot_chol: f64,
    current_smoker: f64,
    bmi: f64,
    target: Option<f64>,
}

// ------------------------------------------------------------
// 3. CHARGEMENT DES DONNÉES PAR DÉFAUT (FRAMINGHAM SIMULÉ)
// ------------------------------------------------------------
fn load_default_data() -> (DMatrix<f64>, DVector<f64>) {
    // Jeu de données synthétique mais réaliste (200 lignes pour démonstration)
    let mut rng = rand::thread_rng();
    let n = 200;
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..n {
        let age = 30.0 + rng.gen::<f64>() * 50.0;
        let sys_bp = 90.0 + rng.gen::<f64>() * 70.0;
        let tot_chol = 150.0 + rng.gen::<f64>() * 150.0;
        let smoker = if rng.gen::<f64>() < 0.3 { 1.0 } else { 0.0 };
        let bmi = 18.0 + rng.gen::<f64>() * 15.0;
        let risk = if age > 60.0 && sys_bp > 140.0 && tot_chol > 200.0 { 0.7 } else { 0.2 };
        let target = if rng.gen::<f64>() < risk { 1.0 } else { 0.0 };
        features.push(vec![age, sys_bp, tot_chol, smoker, bmi]);
        targets.push(target);
    }
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let x = DMatrix::from_row_slice(n, 5, &flat);
    let y = DVector::from_vec(targets);
    (x, y)
}

// ------------------------------------------------------------
// 4. VALIDATION CROISÉE K-FOLD AVEC AUC
// ------------------------------------------------------------
fn kfold_cross_validation(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    k: usize,
    learning_rate: f64,
    epochs: usize,
) -> (Vec<f64>, f64) {
    let n = x.nrows();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);
    let fold_size = n / k;
    let mut auc_scores = Vec::new();

    for fold in 0..k {
        let start = fold * fold_size;
        let end = if fold == k - 1 { n } else { start + fold_size };
        let test_indices = &indices[start..end];
        let train_indices: Vec<usize> = indices.iter().filter(|&&i| !test_indices.contains(&i)).copied().collect();

        let mut x_train = Vec::new();
        let mut y_train = Vec::new();
        for &i in &train_indices {
            for j in 0..x.ncols() { x_train.push(x[(i, j)]); }
            y_train.push(y[i]);
        }
        let x_train_mat = DMatrix::from_row_slice(train_indices.len(), x.ncols(), &x_train);
        let y_train_vec = DVector::from_vec(y_train);

        let mut model = LogisticRegression::new(x.ncols());
        model.train(&x_train_mat, &y_train_vec, learning_rate, epochs);

        let mut predictions = Vec::new();
        let mut labels = Vec::new();
        for &i in test_indices {
            let row = x.row(i).transpose();
            let prob = model.predict_prob(&row);
            predictions.push(prob);
            labels.push(y[i]);
        }
        let auc = compute_auc(&predictions, &labels);
        auc_scores.push(auc);
    }
    let mean_auc = auc_scores.iter().sum::<f64>() / auc_scores.len() as f64;
    (auc_scores, mean_auc)
}

fn compute_auc(predictions: &[f64], labels: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, f64)> = predictions.iter().zip(labels.iter()).map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let pos = labels.iter().filter(|&&l| l == 1.0).count() as f64;
    let neg = labels.iter().filter(|&&l| l == 0.0).count() as f64;
    if pos == 0.0 || neg == 0.0 { return 0.5; }
    let mut tp = 0.0; let mut fp = 0.0; let mut prev_tp = 0.0; let mut prev_fp = 0.0;
    let mut auc = 0.0;
    for &(_, label) in &pairs {
        if label == 1.0 { tp += 1.0; } else { fp += 1.0; }
        let tpr = tp / pos;
        let fpr = fp / neg;
        auc += (tpr + prev_tp) * (fpr - prev_fp) / 2.0;
        prev_tp = tpr; prev_fp = fpr;
    }
    auc
}

// ------------------------------------------------------------
// 5. MATRICE DE CONFUSION ET MÉTRIQUES
// ------------------------------------------------------------
#[derive(Debug, Clone)]
struct ConfusionMatrix {
    tp: u32, tn: u32, fp: u32, fn: u32,
}
impl ConfusionMatrix {
    fn from_predictions(predictions: &[u8], labels: &[f64]) -> Self {
        let mut cm = Self { tp: 0, tn: 0, fp: 0, fn: 0 };
        for (pred, &label) in predictions.iter().zip(labels) {
            let label_u8 = label as u8;
            if *pred == 1 && label_u8 == 1 { cm.tp += 1; }
            else if *pred == 0 && label_u8 == 0 { cm.tn += 1; }
            else if *pred == 1 && label_u8 == 0 { cm.fp += 1; }
            else { cm.fn += 1; }
        }
        cm
    }
    fn accuracy(&self) -> f64 { (self.tp + self.tn) as f64 / (self.tp + self.tn + self.fp + self.fn) as f64 }
    fn sensitivity(&self) -> f64 { self.tp as f64 / (self.tp + self.fn) as f64 }
    fn specificity(&self) -> f64 { self.tn as f64 / (self.tn + self.fp) as f64 }
    fn ppv(&self) -> f64 { self.tp as f64 / (self.tp + self.fp) as f64 }
    fn npv(&self) -> f64 { self.tn as f64 / (self.tn + self.fn) as f64 }
    fn f1_score(&self) -> f64 {
        let prec = self.ppv();
        let rec = self.sensitivity();
        if prec + rec == 0.0 { 0.0 } else { 2.0 * (prec * rec) / (prec + rec) }
    }
}

// ------------------------------------------------------------
// 6. GÉNÉRATION DE COURBE ROC (SVG)
// ------------------------------------------------------------
fn plot_roc_curve(predictions: &[f64], labels: &[f64]) -> Result<String, Box<dyn Error>> {
    let mut pairs: Vec<(f64, f64)> = predictions.iter().zip(labels.iter()).map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let pos = labels.iter().filter(|&&l| l == 1.0).count() as f64;
    let neg = labels.iter().filter(|&&l| l == 0.0).count() as f64;
    if pos == 0.0 || neg == 0.0 { return Ok("<svg></svg>".to_string()); }
    let mut points = vec![(0.0, 0.0)];
    let mut tp = 0.0; let mut fp = 0.0;
    for &(_, label) in &pairs {
        if label == 1.0 { tp += 1.0; } else { fp += 1.0; }
        points.push((fp / neg, tp / pos));
    }
    points.push((1.0, 1.0));

    let mut buffer = vec![];
    {
        let root = SVGBackend::new(&mut buffer, (600, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Courbe ROC", ("sans-serif", 25))
            .margin(20)
            .x_label_area_size(50)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;
        chart.configure_mesh()
            .x_desc("Taux de faux positifs (1 - Spécificité)")
            .y_desc("Taux de vrais positifs (Sensibilité)")
            .draw()?;
        chart.draw_series(LineSeries::new(points, &BLUE))?;
        chart.draw_series(PointSeries::of_element(points, 3, &RED, &|c, s, st| EmptyElement::at(c) + Circle::new((0,0), s, st)))?;
        root.present()?;
    }
    Ok(String::from_utf8(buffer)?)
}

// ------------------------------------------------------------
// 7. OPTIMISATION HYPERPARAMÈTRES (RECHERCHE EN GRILLE)
// ------------------------------------------------------------
fn grid_search(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    k: usize,
) -> (f64, usize, f64) {
    let learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1];
    let epochs_list = [500, 1000, 2000];
    let mut best_auc = 0.0;
    let mut best_lr = 0.01;
    let mut best_epochs = 1000;

    for &lr in &learning_rates {
        for &ep in &epochs_list {
            let (_, mean_auc) = kfold_cross_validation(x, y, k, lr, ep);
            info!("LR={}, epochs={}, AUC moyen={:.4}", lr, ep, mean_auc);
            if mean_auc > best_auc {
                best_auc = mean_auc;
                best_lr = lr;
                best_epochs = ep;
            }
        }
    }
    (best_lr, best_epochs, best_auc)
}

// ------------------------------------------------------------
// 8. GÉNÉRATION DE RAPPORT HTML (PERSONNALISABLE)
// ------------------------------------------------------------
fn generate_html_report(
    metrics: &HashMap<String, f64>,
    auc: f64,
    coefs: &[f64],
    feature_names: &[&str],
) -> String {
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n");
    html.push_str("<title>BAKOME CardioRisk – Rapport d'évaluation</title>\n");
    html.push_str("<style>");
    html.push_str("body{font-family:sans-serif;margin:2em;background:#f0f2f5;}");
    html.push_str(".container{max-width:1000px;margin:auto;background:white;padding:2em;border-radius:1em;box-shadow:0 0 10px rgba(0,0,0,0.1);}");
    html.push_str("h1{color:#007bff;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ccc;padding:8px;text-align:left;}");
    html.push_str(".badge{background:#007bff;color:white;padding:3px 8px;border-radius:12px;font-size:0.8em;}");
    html.push_str("</style></head><body><div class=\"container\">\n");
    html.push_str("<h1>❤️ Rapport d'évaluation du modèle cardiovasculaire</h1>\n");
    html.push_str(&format!("<p>AUC moyen (validation croisée) : <b>{:.3}</b></p>\n", auc));
    html.push_str("<h2>Métriques sur l'ensemble d'entraînement</h2>\n");
    html.push_str("<table><tr><th>Métrique</th><th>Valeur</th></tr>\n");
    for (name, val) in metrics {
        html.push_str(&format!("<tr><td>{}</td><td>{:.3}</td></tr>\n", name, val));
    }
    html.push_str("</table>\n<h2>Coefficients du modèle</h2><ul>\n");
    for (i, &coef) in coefs.iter().enumerate() {
        html.push_str(&format!("<li><b>{}</b> : {:.4}</li>\n", feature_names[i], coef));
    }
    html.push_str("</ul>\n<p><i>Rapport généré par BAKOME CardioRisk Pro – open source, à usage médical et de recherche.</i></p>\n");
    html.push_str("</div></body></html>");
    html
}

// ------------------------------------------------------------
// 9. ÉTAT PARTAGÉ (MODÈLE, DONNÉES, MÉTADONNÉES)
// ------------------------------------------------------------
struct AppState {
    model: LogisticRegression,
    x: DMatrix<f64>,
    y: DVector<f64>,
    feature_names: Vec<String>,
    best_lr: f64,
    best_epochs: usize,
    best_auc: f64,
}

// ------------------------------------------------------------
// 10. ENDPOINTS HTTP (AXUM)
// ------------------------------------------------------------
#[derive(Deserialize)]
struct PredictRequest {
    age: f64,
    systolic_bp: f64,
    cholesterol: f64,
    smoker: f64,
    bmi: f64,
}
#[derive(Serialize)]
struct PredictResponse {
    probability: f64,
    risk_category: String,
}

async fn predict_handler(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<PredictRequest>,
) -> impl IntoResponse {
    let state = state.lock().await;
    let features = DVector::from_vec(vec![req.age, req.systolic_bp, req.cholesterol, req.smoker, req.bmi]);
    let prob = state.model.predict_prob(&features);
    let category = if prob < 0.2 { "Faible" } else if prob < 0.4 { "Modéré" } else if prob < 0.6 { "Élevé" } else { "Très élevé" };
    Json(PredictResponse { probability: prob, risk_category: category.to_string() })
}

async fn metrics_handler(State(state): State<Arc<Mutex<AppState>>>) -> impl IntoResponse {
    let state = state.lock().await;
    let mut predictions = Vec::new();
    let mut labels = Vec::new();
    for i in 0..state.x.nrows() {
        let row = state.x.row(i).transpose();
        predictions.push(state.model.predict_prob(&row));
        labels.push(state.y[i]);
    }
    let auc = compute_auc(&predictions, &labels);
    let mut cm = ConfusionMatrix::from_predictions(&predictions.iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect::<Vec<_>>(), &labels);
    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), cm.accuracy());
    metrics.insert("sensitivity".to_string(), cm.sensitivity());
    metrics.insert("specificity".to_string(), cm.specificity());
    metrics.insert("ppv".to_string(), cm.ppv());
    metrics.insert("npv".to_string(), cm.npv());
    metrics.insert("f1_score".to_string(), cm.f1_score());
    metrics.insert("auc".to_string(), auc);
    Json(metrics)
}

async fn roc_svg_handler(State(state): State<Arc<Mutex<AppState>>>) -> impl IntoResponse {
    let state = state.lock().await;
    let mut predictions = Vec::new();
    let mut labels = Vec::new();
    for i in 0..state.x.nrows() {
        predictions.push(state.model.predict_prob(&state.x.row(i).transpose()));
        labels.push(state.y[i]);
    }
    match plot_roc_curve(&predictions, &labels) {
        Ok(svg) => Html(svg).into_response(),
        Err(e) => (format!("Erreur: {}", e)).into_response(),
    }
}

async fn report_handler(State(state): State<Arc<Mutex<AppState>>>) -> impl IntoResponse {
    let state = state.lock().await;
    let mut predictions = Vec::new();
    let mut labels = Vec::new();
    for i in 0..state.x.nrows() {
        predictions.push(state.model.predict_prob(&state.x.row(i).transpose()));
        labels.push(state.y[i]);
    }
    let auc = compute_auc(&predictions, &labels);
    let cm = ConfusionMatrix::from_predictions(&predictions.iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect::<Vec<_>>(), &labels);
    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), cm.accuracy());
    metrics.insert("sensitivity".to_string(), cm.sensitivity());
    metrics.insert("specificity".to_string(), cm.specificity());
    metrics.insert("ppv".to_string(), cm.ppv());
    metrics.insert("npv".to_string(), cm.npv());
    metrics.insert("f1_score".to_string(), cm.f1_score());
    let coefs = state.model.coefficients_vec();
    let feature_names: Vec<&str> = state.feature_names.iter().map(|s| s.as_str()).collect();
    let html = generate_html_report(&metrics, auc, &coefs, &feature_names);
    Html(html).into_response()
}

async fn upload_handler(
    mut multipart: Multipart,
    State(state): State<Arc<Mutex<AppState>>>,
) -> impl IntoResponse {
    while let Some(field) = multipart.next_field().await.unwrap() {
        if field.name() == Some("file") {
            let data = field.text().await.unwrap();
            if let Ok((x, y, feature_names)) = load_csv_data(&data) {
                let mut state = state.lock().await;
                // Optimisation hyperparamètres
                let (best_lr, best_epochs, best_auc) = grid_search(&x, &y, 5);
                let mut model = LogisticRegression::new(x.ncols());
                model.train(&x, &y, best_lr, best_epochs);
                state.model = model;
                state.x = x;
                state.y = y;
                state.feature_names = feature_names;
                state.best_lr = best_lr;
                state.best_epochs = best_epochs;
                state.best_auc = best_auc;
                info!("Nouveau modèle entraîné avec AUC={:.3}", best_auc);
                return "OK - Modèle entraîné avec succès".into_response();
            } else {
                return "Erreur: CSV invalide (colonnes requises: age,sys_bp,tot_chol,current_smoker,bmi,target)".into_response();
            }
        }
    }
    "Erreur: aucun fichier reçu".into_response()
}

fn load_csv_data(csv_data: &str) -> Result<(DMatrix<f64>, DVector<f64>, Vec<String>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(csv_data.as_bytes());
    let headers = rdr.headers()?.clone();
    let feature_names = vec![
        "age".to_string(), "sys_bp".to_string(), "tot_chol".to_string(),
        "current_smoker".to_string(), "bmi".to_string()
    ];
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for result in rdr.deserialize() {
        let record: PatientRecord = result?;
        features.push(vec![record.age, record.sys_bp, record.tot_chol, record.current_smoker, record.bmi]);
        if let Some(t) = record.target {
            targets.push(t);
        } else {
            return Err("Missing target column".into());
        }
    }
    if features.is_empty() { return Err("No data".into()); }
    let n = features.len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let x = DMatrix::from_row_slice(n, 5, &flat);
    let y = DVector::from_vec(targets);
    Ok((x, y, feature_names))
}

async fn download_model_handler(State(state): State<Arc<Mutex<AppState>>>) -> impl IntoResponse {
    let
