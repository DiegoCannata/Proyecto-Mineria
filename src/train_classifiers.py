def tune_model_random(model, param_dist, X, y, n_iter=25):
  '''Función para encontrar los mejores hiperparámetros de los modelos'''
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        n_iter=n_iter,
        random_state=42
    )
    random_search.fit(X, y)
    print(f"Mejores parámetros para {model} son {random_search.best_params_} y su mejor accuracy es {random_search.best_score_:.2f}")
    return random_search.best_estimator_

def get_models():
  '''Función para obtener el modelo junto con su nombre y guardarlos en una lista'''
    models, names = list(), list()

    # Regresión Logística
    models.append(LogisticRegression())
    names.append('LR')

    # Naive Bayes
    models.append(GaussianNB())
    names.append('NB')

    # Árbol de Decisión
    models.append(DecisionTreeClassifier())
    names.append('AD')

    # Support Vector Machine
    models.append(SVC())
    names.append('SVM')

    # XGBoost multiclass
    models.append(XGBClassifier(
        objective='multi:softmax',
        num_class=6,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    ))
    names.append('XGB')

    # K‑Nearest Neighbors
    models.append(KNeighborsClassifier())
    names.append('KNN')

    # Random Forest
    models.append(RandomForestClassifier())
    names.append('RFC')

    return models, names
