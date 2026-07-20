package decision

import "math"

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Predict computes p = sigmoid(intercept + sum(coef_i * x_i)) on RAW,
// UNSCALED feature values. Confirmed from train.py: fit_logistic is plain
// sklearn.linear_model.LogisticRegression().fit(X, y) with no
// Pipeline/StandardScaler, so there is no normalization step to port here.
// Do NOT add one -- it would silently break inference, since the ported
// coefficients were fit against raw values.
func (m *Model) Predict(features map[string]float64) float64 {
	z := m.Intercept
	for i, name := range m.FeatureOrder {
		z += m.Coefficients[i] * features[name]
	}
	return sigmoid(z)
}
