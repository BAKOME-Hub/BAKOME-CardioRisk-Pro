```markdown
# BAKOME CardioRisk Pro

Cardiovascular risk prediction via logistic regression – Open source medical tool (Rust).

## Features

- Logistic regression with gradient descent
- K‑fold cross‑validation (AUC, sensitivity, specificity)
- Hyperparameter grid search
- Interactive ROC curve (SVG)
- Confusion matrix and full metrics (F1, PPV, NPV)
- Modern web dashboard (Axum)
- CSV import for custom datasets
- HTML report generation
- Binary model export/import

## Quick start

```bash
git clone https://github.com/BAKOME-Hub/BAKOME-CardioRisk-Pro.git
cd BAKOME-CardioRisk-Pro
cargo build --release
cargo run --release
```

Open http://localhost:3001

Default dataset

Synthetic dataset (200 patients, Framingham‑like).
Format: age,sys_bp,tot_chol,current_smoker,bmi,target

Tech stack

· Rust + Axum (backend)
· nalgebra (linear algebra)
· plotters (ROC curve)
· HTML/CSS (frontend)

License

MIT

```
