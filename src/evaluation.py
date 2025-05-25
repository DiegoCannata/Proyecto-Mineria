def plot_macro_roc(y_true_bin, y_score, title, ax=None):
"""Grafica y retorna la curva ROC macro-average."""
    classes = range(y_true_bin.shape[1])
    fpr = dict()
    tpr = dict()
    for i in classes:
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    # Interpola para obtener la curva macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, label=f"{title} (AUC={auc_macro:.3f})")
    ax.plot([0,1], [0,1], '--', color='gray')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC macro‑average\n{title}")
    ax.legend(loc="lower right")
    return ax, auc_macro

def evaluate_model(X, y, model, test_size=0.2, random_state=42):
  """
    - Entrena y evalúa un modelo
    - Imprime classification report y confusion matrix
    - Muestra curva ROC macro
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Scores/probs para ROC
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    target_names = [str(c) for c in np.unique(y)]
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print(f"--- Reporte de Clasificación de {model.__class__.__name__} ---")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    print(f"--- Confusion Matrix de {model.__class__.__name__} ---")
    print(cm)

    # Plot ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    fig, ax = plt.subplots(figsize=(6,6))
    plot_macro_roc(y_test_bin, y_score, model.__class__.__name__, ax=ax)
    plt.show()

    return report, cm
